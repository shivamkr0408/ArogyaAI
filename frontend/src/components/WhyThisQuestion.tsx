/**
 * "Why am I asking this?" -- the information gain behind a follow-up.
 *
 * The question the system asks is chosen to maximise expected reduction in
 * the entropy of the posterior over conditions. That is a real number, so
 * it is shown rather than described: a reader can check that the question
 * being asked is the one that actually separates the candidates, instead of
 * taking it on trust.
 */

import { useState } from 'react';

export default function WhyThisQuestion({
  informationGain,
  rationale,
}: {
  informationGain?: number | null;
  rationale?: string | null;
}) {
  const [open, setOpen] = useState(false);
  if (informationGain == null && !rationale) return null;

  return (
    <div className="ml-1 mt-1">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-400 transition hover:text-brand-600 dark:text-gray-500 dark:hover:text-brand-300"
      >
        <span aria-hidden="true">💡</span>
        Why am I asking this?
        {informationGain != null && (
          <span className="rounded bg-brand-50 px-1.5 py-0.5 font-mono text-[10px] text-brand-700 dark:bg-brand-900/30 dark:text-brand-300">
            {informationGain.toFixed(2)} bits
          </span>
        )}
      </button>

      {open && (
        <p className="mt-1.5 max-w-prose rounded-xl border border-gray-100 bg-white/70 px-3 py-2 text-[11px] leading-relaxed text-gray-600 dark:border-slate-700 dark:bg-slate-800/50 dark:text-gray-300">
          {rationale}
        </p>
      )}
    </div>
  );
}
