/**
 * Why the model picked this condition -- the exact Naive Bayes decomposition.
 *
 * The weights are log-odds, so they are signed: positive is evidence for the
 * condition, negative against. That is a diverging quantity, so it gets the
 * validated diverging pair around a neutral zero line rather than a one-hue
 * magnitude ramp, and the sign is stated in words in the legend and in every
 * tooltip so nothing rests on colour alone.
 */

import type { Explanation } from '../api/client';
import { DivergingBars } from './viz/Bars';
import { dec } from './viz/tokens';

export default function ExplanationCard({ explanation }: { explanation: Explanation }) {
  const contributions = explanation.contributions ?? [];
  if (!contributions.length) return null;

  const hasRunnerUp = Boolean(explanation.compared_with);
  const discriminating = (explanation.top_discriminating ?? []).filter(
    (c) => c.discrimination !== null && c.discrimination !== undefined,
  );

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white/70 p-3 dark:border-slate-700 dark:bg-slate-800/40">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Why this condition
        </h4>
        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600 dark:bg-slate-900/60 dark:text-slate-300">
          {explanation.model}
        </span>
      </div>

      <p className="mt-1 text-[11px] leading-snug text-slate-500 dark:text-slate-400">
        Naive Bayes decomposes exactly, so these are the model's own numbers rather than an
        approximation. Each weight is the {explanation.model === 'BernoulliNB' ? 'log-odds' : 'log-likelihood'}{' '}
        of that symptom for <strong className="font-semibold">{explanation.condition}</strong>.
      </p>

      <div className="mt-3">
        <DivergingBars
          rows={contributions.map((c) => ({
            label: c.label,
            value: c.log_likelihood,
            note: `${(c.support_share * 100).toFixed(0)}% of the reported evidence`,
          }))}
          positiveLabel={`supports ${explanation.condition}`}
          negativeLabel="argues against it"
        />
      </div>

      {hasRunnerUp && discriminating.length > 0 && (
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <h5 className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            What separates it from {explanation.compared_with}
          </h5>
          <p className="mt-0.5 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
            The same weight minus the runner-up's. Near zero means the symptom is common to both and
            does not actually distinguish them — which is worth knowing, and is why a plain list of
            "key symptoms" can mislead.
          </p>
          <div className="mt-2.5">
            <DivergingBars
              rows={discriminating.map((c) => ({
                label: c.label,
                value: c.discrimination as number,
              }))}
              positiveLabel={explanation.condition}
              negativeLabel={explanation.compared_with as string}
            />
          </div>
        </div>
      )}

      {explanation.denial_evidence?.length > 0 && (
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <h5 className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            What you ruled out also counted
          </h5>
          <p className="mt-0.5 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
            Bernoulli Naive Bayes scores symptom absence as well as presence, so a symptom you
            denied is real evidence — not just an omission.
          </p>
          <ul className="mt-2 space-y-1">
            {explanation.denial_evidence.map((d) => (
              <li key={d.label} className="flex items-baseline justify-between gap-3 text-xs">
                <span className="text-slate-600 line-through decoration-1 dark:text-slate-400">
                  {d.label}
                </span>
                <span className="font-mono tabular-nums text-slate-500 dark:text-slate-400">
                  {d.discrimination !== null && d.discrimination !== undefined
                    ? `${d.discrimination >= 0 ? '+' : ''}${dec(d.discrimination, 2)}`
                    : dec(d.log_absence_likelihood, 2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {explanation.expected_but_not_reported?.length > 0 && (
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <h5 className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Common in this condition, but you didn't mention
          </h5>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {explanation.expected_but_not_reported.map((m) => (
              <span
                key={m.label}
                className="rounded-full border border-slate-300 px-2 py-0.5 text-[11px] text-slate-600 dark:border-slate-600 dark:text-slate-400"
              >
                {m.label}
              </span>
            ))}
          </div>
          <p className="mt-1.5 text-[11px] text-slate-400 dark:text-slate-500">
            If any of these do apply, mention them — they would sharpen the assessment.
          </p>
        </div>
      )}
    </div>
  );
}
