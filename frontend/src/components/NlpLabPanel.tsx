/**
 * NLP Lab -- run any sentence through the pipeline and see every stage.
 *
 * This exists so the pipeline can be *shown* rather than described: paste a
 * sentence, and the panel displays the normalization rewrites, the Porter
 * stems, the recognized entities with the strategy that found each one, the
 * ConText cues with their scope, the normalized temporal values, the intent
 * posterior and the distress score -- all from one /api/nlp/analyze call.
 */

import { useEffect, useState } from 'react';
import { api, type NlpAnalysis, type NlpCapabilities } from '../api/client';
import EntityHighlight, { EntityLegend, entityTooltip } from './EntityHighlight';
import { StatTile } from './viz/Bars';

const EXAMPLES: { label: string; text: string; shows: string }[] = [
  {
    label: 'Negation + contrast',
    text: "I don't have any fever but my chest hurts badly since 2 days",
    shows: 'NegEx scope stops at "but", so only the fever is ruled out',
  },
  {
    label: 'Code-mixed Hinglish',
    text: 'mujhe tez bukhar aur badan dard hai teen din se',
    shows: 'Romanized Hindi rewritten to clinical English, spans mapped back',
  },
  {
    label: 'Family history',
    text: 'my father has diabetes and I had dengue last year, now a mild cough',
    shows: 'ConText experiencer + temporality axes separate three mentions',
  },
  {
    label: 'Hypothetical',
    text: 'I am scared I might have dengue, what if I get a high fever',
    shows: 'Nothing reaches the classifier; distress is detected instead',
  },
  {
    label: 'Misspellings',
    text: 'stomuch ake, vomitting and feaver since yesturday',
    shows: 'Bounded edit distance against the medical vocabulary only',
  },
  {
    label: 'Devanagari',
    text: 'बुखार और खांसी, पेट दर्द भी',
    shows: 'Devanagari handled by the same normalization pass',
  },
];

const STRATEGY_NOTE: Record<string, string> = {
  exact: 'dictionary longest-match',
  fuzzy: 'approximate string match',
  embedding: 'MiniLM semantic match',
  lexicon: 'modifier lexicon',
  rule: 'rule-based',
};

const CUE_TONE: Record<string, string> = {
  negation: 'bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-100',
  pseudo: 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200',
  termination: 'bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-100',
  hypothetical: 'bg-indigo-100 text-indigo-900 dark:bg-indigo-900/40 dark:text-indigo-100',
  historical: 'bg-teal-100 text-teal-900 dark:bg-teal-900/40 dark:text-teal-100',
  family: 'bg-violet-100 text-violet-900 dark:bg-violet-900/40 dark:text-violet-100',
  uncertainty: 'bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-100',
};

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white/60 p-3 dark:border-slate-700 dark:bg-slate-800/40">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {title}
      </h4>
      {hint && <p className="mt-0.5 text-[11px] text-slate-400 dark:text-slate-500">{hint}</p>}
      <div className="mt-2">{children}</div>
    </section>
  );
}

export default function NlpLabPanel({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [analysis, setAnalysis] = useState<NlpAnalysis | null>(null);
  const [caps, setCaps] = useState<NlpCapabilities | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [semantic, setSemantic] = useState(false);

  async function run(value: string, useSemantic = semantic) {
    if (!value.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setAnalysis(await api.analyzeText(value, useSemantic));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    run(EXAMPLES[0].text);
    api.nlpCapabilities().then(setCaps).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const symptomEntities = (analysis?.entities ?? []).filter((e) => e.label === 'SYMPTOM');

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/50 p-3 backdrop-blur-sm sm:p-6">
      <div className="w-full max-w-4xl rounded-xl border border-slate-200 bg-slate-50 shadow-soft dark:border-slate-700 dark:bg-slate-900">
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 p-4 dark:border-slate-700">
          <div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">NLP Lab</h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Every stage of the pipeline, run live on any sentence.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-200 dark:text-slate-400 dark:hover:bg-slate-700"
            aria-label="Close NLP Lab"
          >
            Close
          </button>
        </header>

        <div className="space-y-3 p-4">
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                onClick={() => {
                  setText(ex.text);
                  run(ex.text);
                }}
                title={ex.shows}
                className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-[11px] text-slate-700 transition hover:border-brand-400 hover:text-brand-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:hover:text-brand-300"
              >
                {ex.label}
              </button>
            ))}
          </div>

          <div className="flex flex-col gap-2 sm:flex-row">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={2}
              className="flex-1 resize-y rounded-lg border border-slate-300 bg-white p-2 text-sm text-slate-900 outline-none focus:border-brand-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
              placeholder="Type a sentence the way a patient would..."
            />
            <div className="flex flex-col gap-2 sm:w-40">
              <button
                onClick={() => run(text)}
                disabled={loading}
                className="rounded-lg bg-brand-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-brand-700 disabled:opacity-60"
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
              <label className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                <input
                  type="checkbox"
                  checked={semantic}
                  onChange={(e) => {
                    setSemantic(e.target.checked);
                    run(text, e.target.checked);
                  }}
                />
                embedding tier
              </label>
            </div>
          </div>

          {error && (
            <p className="rounded-lg bg-rose-50 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
              {error}
            </p>
          )}

          {analysis && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatTile
                  label="Intent"
                  value={analysis.intent.intent.replace(/_/g, ' ')}
                  sub={`${Math.round(analysis.intent.confidence * 100)}% · ${analysis.intent.source}`}
                />
                <StatTile
                  label="Distress"
                  value={analysis.distress.level}
                  sub={`score ${analysis.distress.score}`}
                  tone={analysis.distress.level === 'high' ? 'warn' : 'neutral'}
                />
                <StatTile
                  label="Entities"
                  value={String(analysis.entities.length)}
                  sub={`${analysis.present_symptoms.length} feed the model`}
                />
                <StatTile label="Pipeline" value={`${analysis.total_ms.toFixed(1)} ms`} sub="all stages" />
              </div>

              <Section
                title="1 · Entities in your original text"
                hint="Spans point at what you typed, even where normalization rewrote the words."
              >
                <p className="text-sm leading-7 text-slate-800 dark:text-slate-200">
                  <EntityHighlight text={analysis.original} entities={analysis.entities} />
                </p>
                <div className="mt-2 border-t border-slate-200 pt-2 dark:border-slate-700">
                  <EntityLegend />
                </div>
              </Section>

              {analysis.normalization_edits.length > 0 && (
                <Section
                  title="2 · Normalization rewrites"
                  hint="Each rewrite keeps a mapping back to the original offsets."
                >
                  <ul className="space-y-1">
                    {analysis.normalization_edits.map((edit, i) => (
                      <li key={i} className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-slate-600 dark:bg-slate-700 dark:text-slate-300">
                          {edit.kind.replace('_', ' ')}
                        </span>
                        <span className="font-mono text-rose-700 line-through dark:text-rose-300">
                          {edit.original}
                        </span>
                        <span className="text-slate-400">to</span>
                        <span className="font-mono text-emerald-700 dark:text-emerald-300">
                          {edit.replacement}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 rounded bg-slate-100 p-2 font-mono text-[11px] text-slate-600 dark:bg-slate-900/60 dark:text-slate-300">
                    {analysis.normalized}
                  </p>
                </Section>
              )}

              {analysis.cues.length > 0 && (
                <Section
                  title="3 · Assertion cues and their scope"
                  hint="NegEx/ConText: each cue governs a bounded window that stops at a termination term."
                >
                  <ul className="space-y-1.5">
                    {analysis.cues.map((cue, i) => (
                      <li key={i} className="flex flex-wrap items-center gap-2 text-xs">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                            CUE_TONE[cue.kind] ?? CUE_TONE.pseudo
                          }`}
                        >
                          {cue.kind}
                        </span>
                        <span className="font-mono text-slate-800 dark:text-slate-200">"{cue.text}"</span>
                        {cue.direction !== 'none' && cue.scope_text && (
                          <>
                            <span className="text-slate-400">{cue.direction === 'forward' ? 'scopes right over' : 'scopes left over'}</span>
                            <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-700 dark:bg-slate-900/60 dark:text-slate-300">
                              {cue.scope_text}
                            </span>
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                </Section>
              )}

              {symptomEntities.length > 0 && (
                <Section
                  title="4 · Symptom recognition detail"
                  hint="Which strategy matched, and what the assertion layer concluded."
                >
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                        <tr>
                          <th className="pb-1 pr-3">Surface</th>
                          <th className="pb-1 pr-3">Canonical</th>
                          <th className="pb-1 pr-3">Strategy</th>
                          <th className="pb-1 pr-3">Assertion</th>
                          <th className="pb-1">Model input</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                        {symptomEntities.map((entity, i) => (
                          <tr key={i} title={entityTooltip(entity)}>
                            <td className="py-1.5 pr-3 font-mono text-slate-800 dark:text-slate-200">
                              {entity.original_text || entity.text}
                            </td>
                            <td className="py-1.5 pr-3 text-slate-600 dark:text-slate-400">
                              {entity.canonical.replace(/_/g, ' ')}
                            </td>
                            <td className="py-1.5 pr-3 text-slate-500 dark:text-slate-400">
                              {STRATEGY_NOTE[entity.strategy] ?? entity.strategy}
                              <span className="ml-1 font-mono text-[10px] text-slate-400">
                                {Math.round(entity.confidence * 100)}%
                              </span>
                            </td>
                            <td className="py-1.5 pr-3">
                              <span className="text-slate-700 dark:text-slate-300">
                                {entity.assertion?.label ?? '-'}
                              </span>
                            </td>
                            <td className="py-1.5">
                              {entity.is_present ? (
                                <span className="text-emerald-700 dark:text-emerald-400">included</span>
                              ) : (
                                <span className="text-slate-400 dark:text-slate-500">excluded</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Section>
              )}

              {(analysis.temporal.length > 0 || analysis.severity) && (
                <Section title="5 · Normalized clinical context" hint="Values the questionnaire no longer has to ask for.">
                  <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                    {analysis.duration_bucket && (
                      <div>
                        <dt className="text-slate-500 dark:text-slate-400">Duration</dt>
                        <dd className="font-medium text-slate-800 dark:text-slate-200">
                          {analysis.duration_bucket}
                          <span className="ml-1 font-mono text-[10px] text-slate-400">
                            ({analysis.duration_days}d)
                          </span>
                        </dd>
                      </div>
                    )}
                    {analysis.trend && (
                      <div>
                        <dt className="text-slate-500 dark:text-slate-400">Trend</dt>
                        <dd className="font-medium text-slate-800 dark:text-slate-200">{analysis.trend}</dd>
                      </div>
                    )}
                    {analysis.frequency && (
                      <div>
                        <dt className="text-slate-500 dark:text-slate-400">Frequency</dt>
                        <dd className="font-medium text-slate-800 dark:text-slate-200">{analysis.frequency}</dd>
                      </div>
                    )}
                    {analysis.severity && (
                      <div>
                        <dt className="text-slate-500 dark:text-slate-400">Severity</dt>
                        <dd className="font-medium text-slate-800 dark:text-slate-200">{analysis.severity}</dd>
                      </div>
                    )}
                  </dl>
                </Section>
              )}

              <Section title="6 · Stage timings" hint="Measured for this sentence, not estimated.">
                <ul className="space-y-1 text-xs">
                  {analysis.timings.map((t) => (
                    <li key={t.name} className="flex items-baseline justify-between gap-3">
                      <span className="text-slate-600 dark:text-slate-400">{t.name}</span>
                      <span className="shrink-0 font-mono tabular-nums text-slate-500 dark:text-slate-400">
                        {t.ms.toFixed(2)} ms
                      </span>
                    </li>
                  ))}
                </ul>
              </Section>

              <details className="rounded-lg border border-slate-200 bg-white/60 p-3 text-xs dark:border-slate-700 dark:bg-slate-800/40">
                <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Tokens and Porter stems ({analysis.tokens.length})
                </summary>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {analysis.tokens.map((tok, i) => (
                    <span
                      key={i}
                      className="rounded border border-slate-200 bg-white px-1.5 py-0.5 font-mono text-[11px] dark:border-slate-600 dark:bg-slate-800"
                      title={`stem: ${tok.stem} · chars ${tok.start}-${tok.end}`}
                    >
                      {tok.text}
                      {tok.stem !== tok.text && (
                        <span className="ml-1 text-slate-400 dark:text-slate-500">→{tok.stem}</span>
                      )}
                    </span>
                  ))}
                </div>
              </details>

              {caps && (
                <details className="rounded-lg border border-slate-200 bg-white/60 p-3 text-xs dark:border-slate-700 dark:bg-slate-800/40">
                  <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    Pipeline stages and lexical resources
                  </summary>
                  <ol className="mt-2 space-y-1.5">
                    {caps.stages.map((stage, i) => (
                      <li key={stage.name}>
                        <span className="font-medium text-slate-800 dark:text-slate-200">
                          {i + 1}. {stage.name}
                        </span>
                        <span className="ml-1 text-slate-500 dark:text-slate-400">
                          — {stage.algorithms.join(', ')}
                        </span>
                      </li>
                    ))}
                  </ol>
                  <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-200 pt-2 sm:grid-cols-3 dark:border-slate-700">
                    {Object.entries(caps.resources).map(([key, value]) => (
                      <div key={key} className="flex items-baseline justify-between gap-2">
                        <dt className="text-slate-500 dark:text-slate-400">{key.replace(/_/g, ' ')}</dt>
                        <dd className="font-mono tabular-nums text-slate-700 dark:text-slate-300">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
