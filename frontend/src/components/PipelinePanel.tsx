import { useState } from 'react';
import type { PipelineStage } from '../api/client';

interface Props {
  stages: PipelineStage[];
}

const STATUS: Record<string, { dot: string; text: string; badge: string }> = {
  ok: {
    dot: 'bg-emerald-500',
    text: 'text-emerald-700 dark:text-emerald-300',
    badge: 'bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-800',
  },
  skipped: {
    dot: 'bg-gray-300 dark:bg-slate-600',
    text: 'text-gray-500 dark:text-gray-400',
    badge: 'bg-gray-50 dark:bg-slate-700/40 border-gray-200 dark:border-slate-600',
  },
  triggered: {
    dot: 'bg-red-500',
    text: 'text-red-700 dark:text-red-300',
    badge: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800',
  },
};

/**
 * Shows exactly which stages ran for the turn the user just sent, and how
 * long each took. This is the "show your working" view — it makes the
 * safety-first ordering and the RAG grounding visible instead of implied.
 */
export default function PipelinePanel({ stages }: Props) {
  const [open, setOpen] = useState(false);
  if (!stages || stages.length === 0) return null;

  const total = stages.reduce((sum, s) => sum + s.ms, 0);

  return (
    <div className="my-2 animate-fade-in-up">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-400 dark:text-gray-500 hover:text-brand-600 dark:hover:text-brand-300 transition"
      >
        <span className="font-mono">{'{}'}</span>
        How this answer was produced · {stages.length} stages · {total.toFixed(0)}ms
        <span className="text-[9px]">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <ol className="mt-2 space-y-1.5 border-l-2 border-dashed border-gray-200 dark:border-slate-700 pl-3.5 ml-1 animate-fade-in-up">
          {stages.map((s, i) => {
            const st = STATUS[s.status] ?? STATUS.ok;
            return (
              <li key={`${s.name}-${i}`} className="relative">
                <span
                  className={`absolute -left-[1.19rem] top-1.5 w-2 h-2 rounded-full ring-2 ring-white dark:ring-slate-900 ${st.dot}`}
                />
                <div className="flex items-baseline justify-between gap-2 flex-wrap">
                  <span className={`text-xs font-semibold ${st.text}`}>{s.name}</span>
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border tabular-nums ${st.badge} ${st.text}`}
                  >
                    {s.status} · {s.ms.toFixed(1)}ms
                  </span>
                </div>
                {s.detail && (
                  <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-snug mt-0.5">
                    {s.detail}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
