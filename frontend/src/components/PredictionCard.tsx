import type { ReactNode } from 'react';

interface Props {
  confidence: number;
  condition: string;
  severity?: string;
  alternatives?: { disease: string; probability: number }[];
  reasoning?: string;
  evidenceSources?: string[];
  showDetails?: boolean;
  onToggle?: () => void;
  disclaimer?: string;
  differential?: ReactNode;
}

const SEVERITY_STYLES: Record<string, { label: string; cls: string; dot: string }> = {
  low: { label: 'Low urgency', cls: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/40 dark:text-emerald-300 dark:border-emerald-800', dot: 'bg-emerald-500' },
  medium: { label: 'Moderate urgency', cls: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/40 dark:text-amber-300 dark:border-amber-800', dot: 'bg-amber-500' },
  high: { label: 'High urgency', cls: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/40 dark:text-red-300 dark:border-red-800', dot: 'bg-red-500' },
};

export default function PredictionCard({
  confidence,
  condition,
  severity = 'medium',
  alternatives,
  reasoning,
  evidenceSources,
  showDetails,
  onToggle,
  disclaimer,
  differential,
}: Props) {
  const percent = Math.round(confidence * 100);
  const sev = SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.medium;

  const barGradient =
    percent >= 70
      ? 'from-emerald-400 to-brand-500'
      : percent >= 50
        ? 'from-yellow-300 to-amber-500'
        : 'from-orange-300 to-red-400';

  const confidenceLabel = percent >= 70 ? 'High' : percent >= 50 ? 'Moderate' : 'Low';

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-100 dark:border-slate-700 shadow-soft p-5 my-2 animate-fade-in-up">
      <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
        <span className="text-xs uppercase tracking-wider text-gray-400 dark:text-gray-500 font-semibold">
          Likely condition
        </span>
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${sev.cls}`}>
          <span className={`w-2 h-2 rounded-full ${sev.dot} animate-pulse`} />
          {sev.label}
        </span>
      </div>

      <h3 className="text-xl font-bold text-gradient mt-1">{condition}</h3>

      <div className="mt-3">
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
          <span>{confidenceLabel} confidence</span>
          <span className="font-semibold text-gray-700 dark:text-gray-200">{percent}%</span>
        </div>
        <div className="w-full h-2.5 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${barGradient} rounded-full transition-all duration-700`}
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      {differential
        ? differential
        : alternatives && alternatives.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-1.5">Other possibilities</p>
              <div className="flex flex-wrap gap-2">
                {alternatives.map((alt) => (
                  <span
                    key={alt.disease}
                    className="text-xs bg-gray-50 dark:bg-slate-700 border border-gray-100 dark:border-slate-600 text-gray-600 dark:text-gray-300 px-2.5 py-1 rounded-full"
                  >
                    {alt.disease} · {Math.round(alt.probability * 100)}%
                  </span>
                ))}
              </div>
            </div>
          )}

      {onToggle && (
        <button
          onClick={onToggle}
          className="mt-3 text-sm text-brand-600 dark:text-brand-300 hover:text-brand-700 font-semibold inline-flex items-center gap-1"
        >
          {showDetails ? 'Hide reasoning ▲' : 'Why this prediction? ▼'}
        </button>
      )}

      {showDetails && reasoning && (
        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-slate-700 text-sm text-gray-600 dark:text-gray-300 space-y-2 animate-fade-in-up">
          <div className="bg-gray-50 dark:bg-slate-700/50 rounded-xl px-3 py-2.5 leading-relaxed">
            {reasoning}
          </div>
          {evidenceSources && evidenceSources.length > 0 && (
            <div>
              <p className="font-medium text-xs text-gray-500 dark:text-gray-400 mb-1">Evidence sources</p>
              <ul className="list-disc list-inside">
                {evidenceSources.filter(Boolean).map((s) => (
                  <li key={s} className="text-xs text-gray-400 dark:text-gray-500 font-mono">{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {disclaimer && (
        <p className="mt-3 text-[11px] leading-relaxed text-gray-400 dark:text-gray-500 border-t border-gray-100 dark:border-slate-700 pt-2">
          ⚠️ {disclaimer}
        </p>
      )}
    </div>
  );
}