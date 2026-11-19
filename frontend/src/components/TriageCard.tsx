import type { Triage } from '../api/client';

interface Props {
  triage: Triage;
  watchFor?: string[];
  clinicalContext?: Record<string, string>;
}

const LEVELS: Record<string, { icon: string; ring: string; head: string; bar: string }> = {
  self_care: {
    icon: '🌱',
    ring: 'border-emerald-200 dark:border-emerald-800 bg-emerald-50/70 dark:bg-emerald-900/20',
    head: 'text-emerald-800 dark:text-emerald-300',
    bar: 'bg-emerald-500',
  },
  soon: {
    icon: '🗓️',
    ring: 'border-amber-200 dark:border-amber-800 bg-amber-50/70 dark:bg-amber-900/20',
    head: 'text-amber-800 dark:text-amber-300',
    bar: 'bg-amber-500',
  },
  urgent: {
    icon: '⚡',
    ring: 'border-red-200 dark:border-red-800 bg-red-50/70 dark:bg-red-900/20',
    head: 'text-red-800 dark:text-red-300',
    bar: 'bg-red-500',
  },
};

/**
 * The actionable half of the result: what to actually do, derived from the
 * condition's baseline severity plus the follow-up answers.
 */
export default function TriageCard({ triage, watchFor, clinicalContext }: Props) {
  const lv = LEVELS[triage.level] ?? LEVELS.soon;

  return (
    <div className={`rounded-2xl border ${lv.ring} p-4 my-2 animate-fade-in-up`}>
      <div className="flex items-start gap-3">
        <span className="text-xl leading-none mt-0.5">{lv.icon}</span>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] uppercase tracking-wider font-semibold text-gray-400 dark:text-gray-500">
            Recommended next step
          </p>
          <h4 className={`font-bold text-sm mt-0.5 ${lv.head}`}>{triage.label}</h4>
          <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 leading-relaxed">
            {triage.detail}
          </p>
        </div>
      </div>

      {clinicalContext && Object.keys(clinicalContext).length > 0 && (
        <div className="mt-3 pt-3 border-t border-black/5 dark:border-white/10">
          <p className="text-[11px] font-semibold text-gray-500 dark:text-gray-400 mb-1.5">
            Based on your answers
          </p>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(clinicalContext).map(([k, v]) => (
              <span
                key={k}
                className="text-[11px] bg-white/70 dark:bg-slate-800/70 border border-black/5 dark:border-white/10 text-gray-600 dark:text-gray-300 px-2 py-0.5 rounded-full"
              >
                <span className="text-gray-400 dark:text-gray-500">{k}:</span> {v}
              </span>
            ))}
          </div>
        </div>
      )}

      {watchFor && watchFor.length > 0 && (
        <div className="mt-3 pt-3 border-t border-black/5 dark:border-white/10">
          <p className="text-[11px] font-semibold text-gray-500 dark:text-gray-400 mb-1.5">
            Seek care promptly if you notice
          </p>
          <ul className="space-y-1">
            {watchFor.map((w) => (
              <li key={w} className="text-xs text-gray-600 dark:text-gray-300 flex gap-2 leading-snug">
                <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${lv.bar}`} />
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
