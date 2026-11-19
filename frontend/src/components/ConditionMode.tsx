/**
 * "I know my condition" mode.
 *
 * The symptom checker answers "what might this be?". This answers the
 * question people ask far more often once they already have a diagnosis:
 * is it contagious, how long will it last, what can I eat, when do I worry.
 *
 * Everything shown here comes from the condition's fact sheet on the
 * backend. There is no free text generated in the browser, so the card
 * cannot claim anything the knowledge base does not hold -- and when a
 * condition is outside the 41 covered, it says so instead of guessing.
 */

import { useEffect, useRef, useState } from 'react';
import {
  conditionApi,
  ConditionFact,
  ConditionSummary,
  ConditionAnswer,
} from '../api/client';

type Turn = { question: string; result: ConditionAnswer };

const URGENCY_STYLE: Record<string, string> = {
  routine: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-800',
  'see a doctor soon': 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-800',
  'needs prompt medical attention': 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800',
};

function Section({ title, children, open = false }: {
  title: string; children: React.ReactNode; open?: boolean;
}) {
  if (!children) return null;
  return (
    <details open={open} className="border-t border-gray-100 py-2 dark:border-slate-700">
      <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {title}
      </summary>
      <div className="mt-1.5 text-sm leading-relaxed text-gray-700 dark:text-gray-200">
        {children}
      </div>
    </details>
  );
}

function Bullets({ items }: { items?: string[] }) {
  if (!items?.length) return null;
  return (
    <ul className="space-y-1">
      {items.map((t) => (
        <li key={t} className="flex gap-2">
          <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-400" />
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );
}

export default function ConditionMode({
  initialText,
  onClose,
  onSwitchToSymptoms,
}: {
  initialText?: string;
  onClose: () => void;
  onSwitchToSymptoms: (seed?: string) => void;
}) {
  const [input, setInput] = useState(initialText ?? '');
  const [loading, setLoading] = useState(false);
  const [fact, setFact] = useState<ConditionFact | null>(null);
  const [emergency, setEmergency] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ConditionSummary[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [chips, setChips] = useState<string[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns, fact]);

  // When we arrive here because the user typed "I have diabetes", resolve it
  // straight away. Making them press Find on text they already typed would
  // be the same mistake as greeting someone who just described a symptom.
  const autoRan = useRef(false);
  useEffect(() => {
    if (initialText && !autoRan.current) {
      autoRan.current = true;
      identify(initialText);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialText]);

  const reset = () => {
    setFact(null); setCandidates([]); setNotice(null);
    setChips([]); setTurns([]); setEmergency(null);
  };

  const identify = async (text: string) => {
    if (!text.trim()) return;
    setLoading(true);
    reset();
    try {
      const r = await conditionApi.match(text.trim());
      if (r.status === 'emergency') { setEmergency(r.message ?? ''); return; }
      if (r.status === 'matched' && r.condition) {
        setFact(r.condition);
        setChips(r.chips ?? []);
        setInput('');   // the box becomes the "ask a question" box from here

        if (r.is_emergency_condition) setEmergency(r.emergency_note ?? null);
        return;
      }
      setCandidates(r.candidates ?? []);
      setNotice(r.message ?? null);
    } catch {
      setNotice('Could not reach the server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const open = async (id: string) => {
    setLoading(true); reset();
    try {
      const r = await conditionApi.get(id);
      if (r.found && r.condition) {
        setFact(r.condition);
        setChips(r.chips ?? []);
        if (r.is_emergency_condition) setEmergency(r.emergency_note ?? null);
      }
    } finally { setLoading(false); }
  };

  const ask = async (question: string) => {
    if (!fact || !question.trim()) return;
    setLoading(true);
    try {
      const r = await conditionApi.ask(fact.id, question.trim());
      if (r.status === 'emergency') { setEmergency(r.message ?? ''); return; }
      setTurns((prev) => [...prev, { question, result: r }]);
      if (r.chips) setChips(r.chips);
    } finally { setLoading(false); }
  };

  return (
    <div className="flex-1 overflow-y-auto overflow-x-hidden">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-extrabold text-gray-800 dark:text-white">
              I know my condition
            </h2>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Ask about a condition you have already been diagnosed with
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-3 py-1.5 text-sm text-gray-500 transition hover:bg-gray-100 dark:hover:bg-slate-700"
          >
            ✕ Close
          </button>
        </div>

        {/* ---- identify ---- */}
        {!fact && (
          <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-soft dark:border-slate-700 dark:bg-slate-800">
            <label className="text-xs font-semibold text-gray-500 dark:text-gray-400">
              Which condition?
            </label>
            <div className="mt-2 flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && identify(input)}
                placeholder='e.g. "diabetes", "sugar", "piles", "TB", "mujhe dengue hai"'
                aria-label="Condition name"
                className="flex-1 rounded-xl border border-gray-200 bg-transparent px-3 py-2 text-sm text-gray-800 placeholder-gray-400 focus:border-brand-400 focus:outline-none dark:border-slate-600 dark:text-gray-100"
              />
              <button
                onClick={() => identify(input)}
                disabled={loading || !input.trim()}
                className="rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105 disabled:from-gray-200 disabled:to-gray-300 disabled:text-gray-400 dark:disabled:from-slate-700 dark:disabled:to-slate-700"
              >
                {loading ? '…' : 'Find'}
              </button>
            </div>
            <p className="mt-2 text-[11px] text-gray-400 dark:text-gray-500">
              Lay names, Hinglish and typos are understood. 41 conditions are covered.
            </p>
          </div>
        )}

        {/* ---- emergency always first ---- */}
        {emergency && (
          <div className="mt-3 rounded-2xl bg-gradient-to-br from-red-600 to-rose-700 p-5 text-white shadow-glow-red">
            <h3 className="text-base font-extrabold tracking-wide">⚠ SEEK URGENT CARE</h3>
            <p className="mt-1 text-sm leading-relaxed">{emergency}</p>
            <p className="mt-2 text-sm font-semibold">
              India: dial{' '}
              <span className="rounded bg-white px-1.5 py-0.5 font-mono text-red-700">108</span>{' '}
              for an ambulance or{' '}
              <span className="rounded bg-white px-1.5 py-0.5 font-mono text-red-700">112</span>.
            </p>
          </div>
        )}

        {/* ---- did you mean / not covered ---- */}
        {notice && (
          <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50/70 p-4 dark:border-amber-800 dark:bg-amber-900/20">
            <p className="text-sm text-amber-900 dark:text-amber-200">{notice}</p>
            {candidates.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {candidates.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => open(c.id)}
                    className="rounded-full border border-amber-300 bg-white/70 px-3 py-1 text-xs font-medium text-amber-900 transition hover:bg-white dark:border-amber-700 dark:bg-slate-800/70 dark:text-amber-200"
                  >
                    {c.name}
                  </button>
                ))}
              </div>
            )}
            <button
              onClick={() => onSwitchToSymptoms(input)}
              className="mt-3 text-xs font-semibold text-brand-600 underline dark:text-brand-300"
            >
              Describe my symptoms instead →
            </button>
          </div>
        )}

        {/* ---- condition card ---- */}
        {fact && (
          <div className="mt-3 rounded-2xl border border-gray-200 bg-white p-5 shadow-soft dark:border-slate-700 dark:bg-slate-800">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-xl font-extrabold text-gray-800 dark:text-white">{fact.name}</h3>
              <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${URGENCY_STYLE[fact.urgency] ?? ''}`}>
                {fact.urgency}
              </span>
              <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${
                fact.is_contagious
                  ? 'border-orange-200 bg-orange-50 text-orange-700 dark:border-orange-800 dark:bg-orange-900/30 dark:text-orange-300'
                  : 'border-gray-200 bg-gray-50 text-gray-600 dark:border-slate-600 dark:bg-slate-700/50 dark:text-gray-300'
              }`}>
                {fact.is_contagious ? 'contagious' : 'not contagious'}
              </span>
              <button
                onClick={reset}
                className="ml-auto text-xs text-gray-400 underline hover:text-brand-500"
              >
                change
              </button>
            </div>

            <p className="mt-3 text-sm leading-relaxed text-gray-700 dark:text-gray-200">
              {fact.overview}
            </p>

            <div className="mt-3">
              <Section title="Common symptoms" open>
                <div className="flex flex-wrap gap-1.5">
                  {fact.common_symptoms.map((s) => (
                    <span key={s} className="rounded-full bg-brand-50 px-2.5 py-0.5 text-xs text-brand-700 dark:bg-brand-900/30 dark:text-brand-300">
                      {s}
                    </span>
                  ))}
                </div>
              </Section>
              <Section title="Causes & risk factors">
                <p>{fact.causes}</p>
                <div className="mt-1.5"><Bullets items={fact.risk_factors} /></div>
              </Section>
              <Section title="Seek care urgently if">
                <Bullets items={fact.condition_specific_red_flags} />
              </Section>
              <Section title="When to see a doctor">{fact.when_to_see_doctor}</Section>
              <Section title="Diet">{fact.diet?.eat}</Section>
              <Section title="Ayurvedic lifestyle">
                <p>{fact.ayurvedic_lifestyle}</p>
                {fact.ayurvedic_caution && (
                  <p className="mt-1.5 rounded-lg bg-amber-50 px-2.5 py-1.5 text-xs text-amber-800 dark:bg-amber-900/30 dark:text-amber-200">
                    ⚠ {fact.ayurvedic_caution}
                  </p>
                )}
              </Section>
              <Section title="Possible complications"><Bullets items={fact.complications} /></Section>
              <Section title="Questions to ask your doctor"><Bullets items={fact.questions_to_ask_doctor} /></Section>
              <Section title="Tests a doctor may use"><Bullets items={fact.common_tests} /></Section>
              <Section title="Sources">
                <div className="flex flex-wrap gap-1.5">
                  {fact.sources.map((s) => (
                    <span key={s} className="rounded-full border border-gray-200 px-2 py-0.5 text-[11px] text-gray-500 dark:border-slate-600 dark:text-gray-400">
                      {s}
                    </span>
                  ))}
                </div>
              </Section>
            </div>

            <button
              onClick={() => onSwitchToSymptoms(`I have ${fact.name} and I have new symptoms`)}
              className="mt-3 w-full rounded-xl border border-brand-200 bg-brand-50/60 px-3 py-2 text-sm font-semibold text-brand-700 transition hover:bg-brand-50 dark:border-brand-800 dark:bg-brand-900/20 dark:text-brand-300"
            >
              I have new symptoms →
            </button>
          </div>
        )}

        {/* ---- Q&A ---- */}
        {turns.map((t, i) => (
          <div key={i} className="mt-3">
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-md bg-gradient-to-br from-indigo-600 to-violet-600 px-4 py-2.5 text-sm text-white">
                {t.question}
              </div>
            </div>
            <div className="mt-2 rounded-2xl rounded-bl-md border border-gray-100 bg-white px-4 py-3 text-sm leading-relaxed text-gray-800 shadow-soft dark:border-slate-700 dark:bg-slate-800 dark:text-gray-100">
              {t.result.answer}
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[10px] text-gray-500 dark:bg-slate-700 dark:text-gray-400">
                  {t.result.intent} · {t.result.intent_source}
                </span>
                {t.result.sections_used?.map((s) => (
                  <span key={s} className="rounded bg-brand-50 px-1.5 py-0.5 font-mono text-[10px] text-brand-700 dark:bg-brand-900/30 dark:text-brand-300">
                    {s}
                  </span>
                ))}
                {t.result.passages?.map((p) => (
                  <span key={p.id} title={p.text} className="rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300">
                    📄 {p.source}
                  </span>
                ))}
              </div>
              <p className="mt-2 text-[10px] text-gray-400 dark:text-gray-500">
                {t.result.disclaimer}
              </p>
            </div>
          </div>
        ))}

        {/* ---- chips + ask box ---- */}
        {fact && (
          <div className="mt-3">
            <div className="flex flex-wrap gap-2">
              {chips.map((c) => (
                <button
                  key={c}
                  onClick={() => (c === 'I have new symptoms'
                    ? onSwitchToSymptoms(`I have ${fact.name} and I have new symptoms`)
                    : ask(c))}
                  disabled={loading}
                  className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 transition hover:border-brand-300 hover:bg-brand-50 disabled:opacity-50 dark:border-slate-600 dark:bg-slate-800 dark:text-gray-200 dark:hover:bg-slate-700"
                >
                  {c}
                </button>
              ))}
            </div>
            <div className="mt-2 flex items-center gap-2 rounded-2xl border border-gray-200 bg-white px-2 py-2 shadow-soft dark:border-slate-600 dark:bg-slate-800">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { ask(input); setInput(''); } }}
                placeholder={`Ask anything about ${fact.name}…`}
                aria-label="Ask about this condition"
                className="flex-1 bg-transparent px-2 text-sm text-gray-800 placeholder-gray-400 focus:outline-none dark:text-gray-100"
              />
              <button
                onClick={() => { ask(input); setInput(''); }}
                disabled={loading || !input.trim()}
                className="rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105 disabled:from-gray-200 disabled:to-gray-300 disabled:text-gray-400 dark:disabled:from-slate-700 dark:disabled:to-slate-700"
              >
                Ask
              </button>
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  );
}
