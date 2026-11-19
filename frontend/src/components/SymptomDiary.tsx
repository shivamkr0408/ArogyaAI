/**
 * Symptom diary with a severity trend per symptom.
 *
 * The chart is a plain line: the job is change over time for one series, so
 * a single hue with a hairline grid is the right form and a categorical
 * palette would be noise. Severity is an ordered 0-3 scale, so the axis is
 * labelled with the words rather than the numbers -- "Moderate" means
 * something to the person, "2" does not.
 *
 * Drawn as inline SVG to avoid adding a charting dependency.
 */

import { useState } from 'react';
import { useDiary, SEVERITY_LABELS, DiaryEntry } from '../hooks/useDiary';

const TREND_STYLE: Record<string, string> = {
  improving: 'text-emerald-600 dark:text-emerald-400',
  worsening: 'text-red-600 dark:text-red-400',
  steady: 'text-gray-500 dark:text-gray-400',
};
const TREND_ICON: Record<string, string> = {
  improving: '↓', worsening: '↑', steady: '→',
};

function shortDate(iso: string) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

function TrendChart({ series }: { series: DiaryEntry[] }) {
  if (series.length === 0) return null;

  const W = 320, H = 96, padL = 56, padR = 24, padT = 8, padB = 20;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const x = (i: number) =>
    padL + (series.length === 1 ? innerW / 2 : (i / (series.length - 1)) * innerW);
  // 3 (severe) at the top, 0 (gone) at the bottom.
  const y = (sev: number) => padT + ((3 - sev) / 3) * innerH;

  const path = series.map((e, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(e.severity).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
         aria-label={`Severity trend over ${series.length} readings`}>
      {/* hairline grid + axis labels */}
      {[3, 2, 1, 0].map((s) => (
        <g key={s}>
          <line x1={padL} x2={W - padR} y1={y(s)} y2={y(s)}
                className="stroke-gray-200 dark:stroke-slate-700" strokeWidth="1" />
          <text x={padL - 6} y={y(s) + 3} textAnchor="end"
                className="fill-gray-400 dark:fill-gray-500" fontSize="8">
            {SEVERITY_LABELS[s]}
          </text>
        </g>
      ))}

      <path d={path} fill="none" strokeWidth="2"
            className="stroke-brand-500" strokeLinecap="round" strokeLinejoin="round" />

      {series.map((e, i) => (
        <circle key={e.date} cx={x(i)} cy={y(e.severity)} r="3"
                className="fill-brand-600 stroke-white dark:stroke-slate-900" strokeWidth="1.5">
          <title>{`${shortDate(e.date)}: ${SEVERITY_LABELS[e.severity]}`}</title>
        </circle>
      ))}

      {/* only label the ends, so the axis never collides with itself */}
      <text x={x(0)} y={H - 6} textAnchor="middle"
            className="fill-gray-400 dark:fill-gray-500" fontSize="8">
        {shortDate(series[0].date)}
      </text>
      {series.length > 1 && (
        <text x={x(series.length - 1)} y={H - 6} textAnchor="middle"
              className="fill-gray-400 dark:fill-gray-500" fontSize="8">
          {shortDate(series[series.length - 1].date)}
        </text>
      )}
    </svg>
  );
}

export default function SymptomDiary({
  onClose,
  suggested = [],
}: {
  onClose: () => void;
  suggested?: string[];
}) {
  const { log, remove, symptoms, seriesFor, trendFor } = useDiary();
  const [name, setName] = useState('');
  const [severity, setSeverity] = useState(1);

  const add = () => {
    const n = name.trim().toLowerCase();
    if (!n) return;
    log(n, severity);
    setName('');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
         onClick={onClose}>
      <div role="dialog" aria-label="Symptom diary"
           className="flex max-h-[88vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl animate-fade-in-up dark:bg-slate-900"
           onClick={(e) => e.stopPropagation()}>
        <header className="flex items-start justify-between border-b border-gray-200 px-5 py-4 dark:border-slate-700">
          <div>
            <h2 className="text-base font-bold text-gray-800 dark:text-white">Symptom diary</h2>
            <p className="mt-0.5 text-[11px] text-gray-500 dark:text-gray-400">
              One reading a day. "Better or worse than last week?" needs more than one point.
            </p>
          </div>
          <button onClick={onClose} aria-label="Close diary"
                  className="rounded-lg px-2 py-1 text-gray-400 transition hover:bg-gray-100 dark:hover:bg-slate-700">
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {/* --- log a reading --- */}
          <div className="rounded-xl border border-gray-200 p-3 dark:border-slate-700">
            <label className="text-xs font-semibold text-gray-500 dark:text-gray-400">
              Log today
            </label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && add()}
              placeholder="symptom, e.g. headache"
              aria-label="Symptom name"
              className="mt-1.5 w-full rounded-lg border border-gray-200 bg-transparent px-2.5 py-1.5 text-sm text-gray-800 placeholder-gray-400 focus:border-brand-400 focus:outline-none dark:border-slate-600 dark:text-gray-100"
            />
            {suggested.length > 0 && !name && (
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {suggested.slice(0, 6).map((s) => (
                  <button key={s} onClick={() => setName(s.replace(/_/g, ' '))}
                          className="rounded-full bg-brand-50 px-2 py-0.5 text-[11px] text-brand-700 dark:bg-brand-900/30 dark:text-brand-300">
                    {s.replace(/_/g, ' ')}
                  </button>
                ))}
              </div>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {SEVERITY_LABELS.map((lbl, i) => (
                <button key={lbl} onClick={() => setSeverity(i)} aria-pressed={severity === i}
                        className={`rounded-full border px-2.5 py-1 text-xs transition ${
                          severity === i
                            ? 'border-brand-500 bg-brand-500 text-white'
                            : 'border-gray-200 text-gray-600 dark:border-slate-600 dark:text-gray-300'
                        }`}>
                  {lbl}
                </button>
              ))}
              <button onClick={add} disabled={!name.trim()}
                      className="ml-auto rounded-lg bg-gradient-to-r from-brand-500 to-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:brightness-105 disabled:from-gray-200 disabled:to-gray-300 disabled:text-gray-400 dark:disabled:from-slate-700 dark:disabled:to-slate-700">
                Save
              </button>
            </div>
          </div>

          {/* --- tracked symptoms --- */}
          {symptoms.length === 0 ? (
            <div className="mt-8 text-center">
              <p className="text-3xl">📈</p>
              <p className="mt-2 text-sm font-medium text-gray-600 dark:text-gray-300">
                Nothing tracked yet
              </p>
              <p className="mx-auto mt-1 max-w-xs text-xs leading-relaxed text-gray-400 dark:text-gray-500">
                Log a symptom each day and the trend appears here. It is included
                in the report you can take to a doctor.
              </p>
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {symptoms.map((s) => {
                const series = seriesFor(s);
                const trend = trendFor(s);
                return (
                  <div key={s} className="rounded-xl border border-gray-200 p-3 dark:border-slate-700">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-100">{s}</h3>
                      <div className="flex items-center gap-2">
                        {trend && (
                          <span className={`text-[11px] font-semibold ${TREND_STYLE[trend]}`}>
                            {TREND_ICON[trend]} {trend}
                          </span>
                        )}
                        <span className="text-[11px] text-gray-400 dark:text-gray-500">
                          {series.length} reading{series.length === 1 ? '' : 's'}
                        </span>
                        <button onClick={() => remove(s)} aria-label={`Stop tracking ${s}`}
                                className="text-gray-300 transition hover:text-red-500 dark:text-slate-600">
                          ✕
                        </button>
                      </div>
                    </div>
                    <TrendChart series={series} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
