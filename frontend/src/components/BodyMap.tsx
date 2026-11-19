/**
 * Point-to-your-symptom picker.
 *
 * Typing is a real barrier here: the user has to name a sensation before the
 * system can help them, and "where it hurts" is much easier to point at than
 * to spell. This is also the accessibility path for anyone who finds the
 * keyboard awkward, alongside the existing voice input.
 *
 * The regions map to phrases the extractor already understands -- the
 * colloquial aliases in backend/app/nlp/lexicon.py -- so a tap produces the
 * same canonical symptoms as typing the sentence would. Picks are composed
 * into one sentence and dropped into the composer rather than sent, so the
 * user can still edit or add detail before sending.
 */

import { useState } from 'react';

interface Region {
  id: string;
  label: string;
  /** Phrasings the NLP layer recognises, offered as chips. */
  symptoms: string[];
}

const REGIONS: Region[] = [
  { id: 'head', label: 'Head', symptoms: ['headache', 'dizziness', 'blurred vision', 'yellow eyes'] },
  { id: 'throat', label: 'Throat & nose', symptoms: ['sore throat', 'runny nose', 'continuous sneezing', 'congestion'] },
  { id: 'chest', label: 'Chest', symptoms: ['chest pain', 'breathlessness', 'cough', 'wheezing', 'palpitations'] },
  { id: 'abdomen', label: 'Stomach', symptoms: ['abdominal pain', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'acidity'] },
  { id: 'back', label: 'Back & neck', symptoms: ['back pain', 'neck pain', 'joint stiffness'] },
  { id: 'arms', label: 'Arms & hands', symptoms: ['joint pain', 'joint swelling', 'weakness in limbs', 'numbness'] },
  { id: 'legs', label: 'Legs & feet', symptoms: ['knee pain', 'swelling', 'numbness', 'slow healing wound'] },
  { id: 'skin', label: 'Skin', symptoms: ['skin rash', 'itching', 'blister', 'skin peeling', 'discoloured patches'] },
  { id: 'general', label: 'All over', symptoms: ['fever', 'chills', 'fatigue', 'body ache', 'weight loss', 'night sweats'] },
];

/** Clickable areas over the figure. Values are percentages of the SVG box. */
const HOTSPOTS: Record<string, { cx: number; cy: number; rx: number; ry: number }> = {
  head: { cx: 50, cy: 9, rx: 9, ry: 8 },
  throat: { cx: 50, cy: 19, rx: 6, ry: 4 },
  chest: { cx: 50, cy: 31, rx: 15, ry: 9 },
  abdomen: { cx: 50, cy: 47, rx: 13, ry: 8 },
  arms: { cx: 76, cy: 38, rx: 7, ry: 17 },
  legs: { cx: 50, cy: 76, rx: 18, ry: 18 },
};

export default function BodyMap({ onCompose }: { onCompose: (text: string) => void }) {
  const [open, setOpen] = useState(false);
  const [region, setRegion] = useState<Region>(REGIONS[0]);
  const [picked, setPicked] = useState<string[]>([]);

  const toggle = (s: string) =>
    setPicked(prev => (prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]));

  const compose = () => {
    if (!picked.length) return;
    const list =
      picked.length === 1
        ? picked[0]
        : `${picked.slice(0, -1).join(', ')} and ${picked[picked.length - 1]}`;
    onCompose(`I have ${list}`);
    setPicked([]);
    setOpen(false);
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="Point to where it hurts instead of typing"
        aria-label="Open body map"
        className="flex-shrink-0 flex h-9 w-9 items-center justify-center rounded-xl text-gray-400 transition hover:bg-brand-50 hover:text-brand-500 dark:hover:bg-slate-700"
      >
        🧍
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <div
            className="flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl animate-fade-in-up dark:bg-slate-900"
            onClick={e => e.stopPropagation()}
          >
            <header className="flex items-center justify-between border-b border-gray-200 px-5 py-3.5 dark:border-slate-700">
              <div>
                <h2 className="text-base font-bold text-gray-800 dark:text-white">
                  Where is the problem?
                </h2>
                <p className="text-[11px] text-gray-500 dark:text-gray-400">
                  Tap a part of the body, then pick what you feel
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                aria-label="Close body map"
                className="rounded-lg px-2 py-1 text-gray-400 transition hover:bg-gray-100 dark:hover:bg-slate-700"
              >
                ✕
              </button>
            </header>

            <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-5 sm:flex-row">
              {/* ---- figure ---- */}
              <div className="flex-shrink-0 sm:w-44">
                <svg viewBox="0 0 100 100" className="mx-auto h-44 w-auto sm:h-56" role="img"
                     aria-label="Simplified body diagram with selectable regions">
                  {/* head */}
                  <circle cx="50" cy="9" r="8" className="fill-gray-200 dark:fill-slate-700" />
                  {/* neck + torso */}
                  <rect x="46" y="16" width="8" height="6" className="fill-gray-200 dark:fill-slate-700" />
                  <path d="M35 23 h30 l3 26 h-36 z" className="fill-gray-200 dark:fill-slate-700" />
                  <path d="M33 49 h34 l-2 12 h-30 z" className="fill-gray-200 dark:fill-slate-700" />
                  {/* arms */}
                  <path d="M35 24 l-9 3 l-4 28 h7 l8 -24 z" className="fill-gray-200 dark:fill-slate-700" />
                  <path d="M65 24 l9 3 l4 28 h-7 l-8 -24 z" className="fill-gray-200 dark:fill-slate-700" />
                  {/* legs */}
                  <path d="M37 61 h11 l-1 33 h-9 z" className="fill-gray-200 dark:fill-slate-700" />
                  <path d="M52 61 h11 l-1 33 h-9 z" className="fill-gray-200 dark:fill-slate-700" />

                  {Object.entries(HOTSPOTS).map(([id, h]) => {
                    const r = REGIONS.find(x => x.id === id)!;
                    const active = region.id === id;
                    return (
                      <ellipse
                        key={id}
                        cx={h.cx} cy={h.cy} rx={h.rx} ry={h.ry}
                        onClick={() => setRegion(r)}
                        className={`cursor-pointer transition ${
                          active
                            ? 'fill-brand-500/45 stroke-brand-600'
                            : 'fill-transparent stroke-transparent hover:fill-brand-400/25'
                        }`}
                        strokeWidth="1"
                      >
                        <title>{r.label}</title>
                      </ellipse>
                    );
                  })}
                </svg>

                <div className="mt-2 flex flex-wrap justify-center gap-1">
                  {REGIONS.filter(r => !HOTSPOTS[r.id]).map(r => (
                    <button
                      key={r.id}
                      onClick={() => setRegion(r)}
                      className={`rounded-full border px-2.5 py-1 text-[11px] transition ${
                        region.id === r.id
                          ? 'border-brand-400 bg-brand-50 text-brand-700 dark:bg-brand-900/30 dark:text-brand-300'
                          : 'border-gray-200 text-gray-500 hover:border-brand-200 dark:border-slate-600 dark:text-gray-400'
                      }`}
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* ---- symptoms for the chosen region ---- */}
              <div className="flex-1">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
                  {region.label}
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {region.symptoms.map(s => {
                    const on = picked.includes(s);
                    return (
                      <button
                        key={s}
                        onClick={() => toggle(s)}
                        aria-pressed={on}
                        className={`rounded-xl border px-3 py-1.5 text-sm transition ${
                          on
                            ? 'border-brand-500 bg-brand-500 text-white shadow-sm'
                            : 'border-gray-200 text-gray-700 hover:border-brand-300 hover:bg-brand-50 dark:border-slate-600 dark:text-gray-200 dark:hover:bg-slate-700'
                        }`}
                      >
                        {on && '✓ '}{s}
                      </button>
                    );
                  })}
                </div>

                {picked.length > 0 && (
                  <div className="mt-4 rounded-xl border border-brand-200 bg-brand-50/60 p-3 dark:border-brand-800 dark:bg-brand-900/20">
                    <p className="text-[11px] font-semibold text-gray-500 dark:text-gray-400">
                      This will be sent as
                    </p>
                    <p className="mt-0.5 text-sm text-gray-800 dark:text-gray-100">
                      “I have {picked.length === 1
                        ? picked[0]
                        : `${picked.slice(0, -1).join(', ')} and ${picked[picked.length - 1]}`}”
                    </p>
                  </div>
                )}
              </div>
            </div>

            <footer className="flex items-center justify-between gap-3 border-t border-gray-200 px-5 py-3 dark:border-slate-700">
              <span className="text-[11px] text-gray-400 dark:text-gray-500">
                {picked.length === 0
                  ? 'Nothing selected yet'
                  : `${picked.length} selected — you can still edit the text before sending`}
              </span>
              <div className="flex gap-2">
                {picked.length > 0 && (
                  <button
                    onClick={() => setPicked([])}
                    className="rounded-xl px-3 py-2 text-sm text-gray-500 transition hover:bg-gray-100 dark:hover:bg-slate-700"
                  >
                    Clear
                  </button>
                )}
                <button
                  onClick={compose}
                  disabled={picked.length === 0}
                  className="rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105 disabled:from-gray-200 disabled:to-gray-300 disabled:text-gray-400 dark:disabled:from-slate-700 dark:disabled:to-slate-700"
                >
                  Add to message
                </button>
              </div>
            </footer>
          </div>
        </div>
      )}
    </>
  );
}
