/**
 * Per-turn NLP readout, collapsed under the reply.
 *
 * The pipeline panel already shows *which* stages ran. This shows what they
 * actually concluded about this specific message: the rewrites, the entities
 * highlighted in the user's own words, what was ruled out, and the intent.
 *
 * Collapsed by default -- it is evidence, not the main content -- but the
 * summary line carries the headline so it is useful without expanding.
 */

import type { NlpTrace } from '../api/client';
import EntityHighlight, { EntityLegend } from './EntityHighlight';

export default function NlpTurnPanel({
  nlp,
  userText,
  deniedSymptoms,
}: {
  nlp: NlpTrace;
  userText: string;
  deniedSymptoms: string[];
}) {
  const symptoms = (nlp.entities ?? []).filter((e) => e.label === 'SYMPTOM');
  const present = symptoms.filter((e) => e.is_present);
  const rewrites = nlp.normalization_edits ?? [];
  const context = [
    nlp.duration_bucket && `duration ${nlp.duration_bucket}`,
    nlp.severity && `${nlp.severity} intensity`,
    nlp.trend && `${nlp.trend}`,
    nlp.frequency && `${nlp.frequency}`,
  ].filter(Boolean) as string[];

  // Nothing worth showing for a bare "Yes" answer.
  if (!symptoms.length && !rewrites.length && !context.length && !deniedSymptoms.length) {
    return null;
  }

  // The summary used to count only symptoms, negations and rewrites, so a
  // turn that produced just context -- answering "Less than a day" yields a
  // duration and nothing else -- rendered a header with no text under it and
  // looked broken. Context now counts towards the summary as well.
  const summary = [
    present.length ? `${present.length} symptom${present.length === 1 ? '' : 's'}` : null,
    deniedSymptoms.length ? `${deniedSymptoms.length} ruled out` : null,
    rewrites.length ? `${rewrites.length} rewrite${rewrites.length === 1 ? '' : 's'}` : null,
    ...context,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <details className="group ml-1 rounded-xl border border-gray-100 bg-white/70 px-3 py-2 text-xs dark:border-slate-700 dark:bg-slate-800/40">
      <summary className="flex cursor-pointer flex-wrap items-center gap-2 text-gray-500 dark:text-gray-400">
        <span className="font-semibold uppercase tracking-wide text-[10px]">Language analysis</span>
        {summary && <span className="text-gray-400 dark:text-gray-500">{summary}</span>}
        <span className="ml-auto font-mono text-[10px] text-gray-400 dark:text-gray-500">
          {nlp.total_ms.toFixed(0)} ms
        </span>
      </summary>

      <div className="mt-2.5 space-y-2.5">
        {userText && symptoms.length > 0 && (
          <div>
            <p className="leading-6 text-gray-700 dark:text-gray-200">
              <EntityHighlight text={userText} entities={nlp.entities} />
            </p>
            <div className="mt-1.5">
              <EntityLegend />
            </div>
          </div>
        )}

        {rewrites.length > 0 && (
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
              Normalized
            </p>
            <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
              {rewrites.map((edit, i) => (
                <li key={i} className="text-gray-600 dark:text-gray-300">
                  <span className="font-mono line-through decoration-rose-400">{edit.original}</span>
                  <span className="mx-1 text-gray-400">→</span>
                  <span className="font-mono text-emerald-700 dark:text-emerald-300">
                    {edit.replacement}
                  </span>
                  <span className="ml-1 text-[10px] text-gray-400 dark:text-gray-500">
                    ({edit.kind.replace('_', ' ')})
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {deniedSymptoms.length > 0 && (
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
              Ruled out by you
            </p>
            <p className="mt-1 text-gray-600 dark:text-gray-300">
              {deniedSymptoms.map((s) => s.replace(/_/g, ' ')).join(', ')}
              <span className="ml-1 text-[10px] text-gray-400 dark:text-gray-500">
                — counted as evidence, not just skipped
              </span>
            </p>
          </div>
        )}

        {context.length > 0 && (
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 dark:text-gray-500">
              Captured from your wording
            </p>
            <p className="mt-1 text-gray-600 dark:text-gray-300">
              {context.join(' · ')}
              <span className="ml-1 text-[10px] text-gray-400 dark:text-gray-500">
                — so you are not asked again
              </span>
            </p>
          </div>
        )}

        {nlp.intent?.intent && (
          <p className="text-[10px] text-gray-400 dark:text-gray-500">
            intent: <span className="font-mono">{nlp.intent.intent}</span> (
            {Math.round(nlp.intent.confidence * 100)}%, {nlp.intent.source})
            {nlp.distress?.level && nlp.distress.level !== 'calm' && (
              <> · distress: {nlp.distress.level}</>
            )}
          </p>
        )}
      </div>
    </details>
  );
}
