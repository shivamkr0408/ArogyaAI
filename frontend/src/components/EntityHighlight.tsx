/**
 * Renders a message with its recognized entities highlighted in place.
 *
 * This is the payoff for the offset-mapping work in the normalizer: the spans
 * point into the text the user actually typed, so "pet dard" lights up even
 * though the pipeline rewrote it to "stomach pain" before recognizing it.
 *
 * Assertion status drives the styling, not just presence: a negated symptom is
 * struck through, a family-attributed one is dashed, a hypothetical is faded.
 * Every mark also carries a text label in its tooltip, so the information is
 * never colour-alone.
 */

import type { Entity } from '../api/client';

const LABEL_STYLES: Record<string, string> = {
  SYMPTOM: 'bg-brand-100 text-brand-900 ring-brand-300 dark:bg-brand-900/50 dark:text-brand-100 dark:ring-brand-700',
  DISEASE: 'bg-violet-100 text-violet-900 ring-violet-300 dark:bg-violet-900/40 dark:text-violet-100 dark:ring-violet-700',
  SEVERITY: 'bg-amber-100 text-amber-900 ring-amber-300 dark:bg-amber-900/40 dark:text-amber-100 dark:ring-amber-700',
  BODY_SITE: 'bg-sky-100 text-sky-900 ring-sky-300 dark:bg-sky-900/40 dark:text-sky-100 dark:ring-sky-700',
  TEMPORAL: 'bg-slate-200 text-slate-800 ring-slate-300 dark:bg-slate-700 dark:text-slate-100 dark:ring-slate-600',
};

const ASSERTION_NOTE: Record<string, string> = {
  negated: 'ruled out by the user',
  family: 'someone else, not the patient',
  hypothetical: 'hypothetical, not reported',
  historical: 'past episode, not current',
  uncertain: 'reported with uncertainty',
  affirmed: 'currently reported',
};

export function entityTooltip(entity: Entity): string {
  const parts = [`${entity.label.toLowerCase()}: ${entity.canonical.replace(/_/g, ' ')}`];
  if (entity.assertion) parts.push(ASSERTION_NOTE[entity.assertion.label] ?? entity.assertion.label);
  parts.push(`matched by ${entity.strategy} (${Math.round(entity.confidence * 100)}%)`);
  if (entity.severity) parts.push(`severity: ${entity.severity}`);
  if (entity.assertion?.cues?.length) {
    parts.push(`cue: "${entity.assertion.cues.map((c) => c.text).join('", "')}"`);
  }
  return parts.join(' · ');
}

function assertionClass(entity: Entity): string {
  const label = entity.assertion?.label;
  if (label === 'negated') return 'line-through decoration-2 opacity-70';
  if (label === 'family') return 'border-b-2 border-dashed';
  if (label === 'hypothetical' || label === 'historical') return 'opacity-60 italic';
  if (label === 'uncertain') return 'border-b border-dotted';
  return '';
}

export default function EntityHighlight({
  text,
  entities,
  className = '',
}: {
  text: string;
  entities: Entity[];
  className?: string;
}) {
  if (!entities?.length) return <span className={className}>{text}</span>;

  // Keep one mark per character range: entities can nest (a body site inside a
  // symptom), and overlapping <mark> elements would break the text flow.
  const sorted = [...entities]
    .filter((e) => e.char_end > e.char_start && e.char_start >= 0 && e.char_end <= text.length)
    .sort((a, b) => a.char_start - b.char_start || b.char_end - a.char_end);

  const chosen: Entity[] = [];
  let claimedTo = -1;
  for (const entity of sorted) {
    if (entity.char_start < claimedTo) continue;
    chosen.push(entity);
    claimedTo = entity.char_end;
  }

  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  chosen.forEach((entity, i) => {
    if (entity.char_start > cursor) nodes.push(text.slice(cursor, entity.char_start));
    const style = LABEL_STYLES[entity.label] ?? LABEL_STYLES.TEMPORAL;
    nodes.push(
      <mark
        key={`${entity.canonical}-${entity.char_start}-${i}`}
        title={entityTooltip(entity)}
        className={`rounded px-1 py-[1px] ring-1 ring-inset ${style} ${assertionClass(entity)}`}
      >
        {text.slice(entity.char_start, entity.char_end)}
      </mark>,
    );
    cursor = entity.char_end;
  });
  if (cursor < text.length) nodes.push(text.slice(cursor));

  return <span className={className}>{nodes}</span>;
}

export function EntityLegend() {
  const items: { label: string; name: string }[] = [
    { label: 'SYMPTOM', name: 'Symptom' },
    { label: 'DISEASE', name: 'Condition' },
    { label: 'SEVERITY', name: 'Severity' },
    { label: 'BODY_SITE', name: 'Body site' },
    { label: 'TEMPORAL', name: 'Time' },
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px]">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <span
            className={`inline-block h-3 w-3 rounded-sm ring-1 ring-inset ${
              LABEL_STYLES[item.label].split(' ').filter((c) => !c.startsWith('text-')).join(' ')
            }`}
          />
          <span className="text-slate-600 dark:text-slate-400">{item.name}</span>
        </span>
      ))}
      <span className="ml-1 text-slate-400 dark:text-slate-500">
        struck through = ruled out · dashed = family history · faded = hypothetical or past
      </span>
    </div>
  );
}
