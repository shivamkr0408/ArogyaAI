/**
 * Two bar primitives shared by the evaluation and explainability panels.
 *
 * Both follow the project's mark spec: thin bars, 4px rounded data-ends
 * anchored to the baseline, a 2px surface gap between adjacent fills,
 * recessive gridlines, selective direct labels (the value on each row, never
 * a number on every tick), and a hover tooltip on every mark.
 */

import { EMPHASIS_OFF, EMPHASIS_ON, POLE_NEGATIVE, POLE_POSITIVE, dec } from './tokens';

export interface EmphasisRow {
  label: string;
  value: number;
  /** The subject of the chart -- gets the accent; everything else is grey. */
  emphasized?: boolean;
  note?: string;
}

/**
 * Horizontal magnitude bars with one row emphasized.
 *
 * Used where the reader's question is "how does the deployed model compare?",
 * which is an emphasis question, not an identity one -- so there is no
 * categorical palette and no legend box; the accent row is direct-labelled.
 */
export function EmphasisBars({
  rows,
  max,
  valueFormat = (v: number) => dec(v),
  caption,
}: {
  rows: EmphasisRow[];
  max?: number;
  valueFormat?: (v: number) => string;
  caption?: string;
}) {
  const ceiling = max ?? Math.max(...rows.map((r) => r.value), 0.0001);

  return (
    <div>
      <ul className="space-y-2">
        {rows.map((row) => {
          const width = Math.max(1, (row.value / ceiling) * 100);
          return (
            <li key={row.label} className="group">
              <div className="flex items-baseline justify-between gap-3 text-xs">
                <span
                  className={
                    row.emphasized
                      ? 'font-semibold text-slate-900 dark:text-slate-100'
                      : 'text-slate-600 dark:text-slate-400'
                  }
                >
                  {row.label}
                  {row.emphasized && (
                    <span className="ml-2 rounded bg-brand-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand-800 dark:bg-brand-900/60 dark:text-brand-200">
                      deployed
                    </span>
                  )}
                </span>
                <span
                  className={
                    row.emphasized
                      ? 'font-mono font-semibold tabular-nums text-slate-900 dark:text-slate-100'
                      : 'font-mono tabular-nums text-slate-500 dark:text-slate-400'
                  }
                >
                  {valueFormat(row.value)}
                </span>
              </div>
              <div
                className="mt-1 h-2 w-full overflow-hidden rounded-sm bg-slate-100 dark:bg-slate-800"
                title={`${row.label}: ${valueFormat(row.value)}${row.note ? ` - ${row.note}` : ''}`}
              >
                <div
                  className={`h-full rounded-r ${row.emphasized ? EMPHASIS_ON : EMPHASIS_OFF}`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
      {caption && (
        <p className="mt-3 text-[11px] leading-snug text-slate-500 dark:text-slate-400">{caption}</p>
      )}
    </div>
  );
}

export interface DivergingRow {
  label: string;
  value: number;
  note?: string;
}

/**
 * Signed bars around a zero baseline.
 *
 * The Naive Bayes weights are log-odds, so they are genuinely bidirectional:
 * positive is evidence *for* the condition, negative is evidence *against*.
 * A sequential ramp would hide that, so this uses the validated diverging
 * pair with a neutral centre line, and labels the sign in words as well as
 * colour so identity never rests on hue alone.
 */
export function DivergingBars({
  rows,
  caption,
  positiveLabel = 'supports',
  negativeLabel = 'argues against',
}: {
  rows: DivergingRow[];
  caption?: string;
  positiveLabel?: string;
  negativeLabel?: string;
}) {
  const extent = Math.max(...rows.map((r) => Math.abs(r.value)), 0.0001);

  return (
    <div>
      <div className="mb-2 flex items-center gap-4 text-[11px] text-slate-500 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className={`inline-block h-2 w-3 rounded-sm ${POLE_POSITIVE}`} />
          {positiveLabel}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className={`inline-block h-2 w-3 rounded-sm ${POLE_NEGATIVE}`} />
          {negativeLabel}
        </span>
      </div>

      <ul className="space-y-2">
        {rows.map((row) => {
          const magnitude = (Math.abs(row.value) / extent) * 50;
          const positive = row.value >= 0;
          return (
            <li key={row.label} className="grid grid-cols-[9rem_1fr_4.5rem] items-center gap-2">
              <span className="truncate text-xs text-slate-600 dark:text-slate-400" title={row.label}>
                {row.label}
              </span>
              <div
                className="relative h-2.5 w-full rounded-sm bg-slate-100 dark:bg-slate-800"
                title={`${row.label}: ${row.value >= 0 ? '+' : ''}${dec(row.value)} (${
                  positive ? positiveLabel : negativeLabel
                })${row.note ? ` - ${row.note}` : ''}`}
              >
                {/* Neutral centre line -- the zero baseline. */}
                <span className="absolute left-1/2 top-[-2px] h-[calc(100%+4px)] w-px bg-slate-300 dark:bg-slate-600" />
                <span
                  className={`absolute top-0 h-full ${positive ? POLE_POSITIVE : POLE_NEGATIVE} ${
                    positive ? 'rounded-r' : 'rounded-l'
                  }`}
                  style={
                    positive
                      ? { left: '50%', width: `${magnitude}%`, marginLeft: '1px' }
                      : { right: '50%', width: `${magnitude}%`, marginRight: '1px' }
                  }
                />
              </div>
              <span
                className={`text-right font-mono text-xs tabular-nums ${
                  positive ? 'text-slate-700 dark:text-slate-300' : 'text-slate-500 dark:text-slate-400'
                }`}
              >
                {row.value >= 0 ? '+' : ''}
                {dec(row.value, 2)}
              </span>
            </li>
          );
        })}
      </ul>
      {caption && (
        <p className="mt-3 text-[11px] leading-snug text-slate-500 dark:text-slate-400">{caption}</p>
      )}
    </div>
  );
}

/** A single headline metric. Not a one-bar bar chart. */
export function StatTile({
  label,
  value,
  sub,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: 'neutral' | 'good' | 'warn';
}) {
  const toneClass =
    tone === 'good'
      ? 'text-emerald-700 dark:text-emerald-400'
      : tone === 'warn'
        ? 'text-amber-700 dark:text-amber-400'
        : 'text-slate-900 dark:text-slate-100';
  return (
    <div className="rounded-lg border border-slate-200 bg-white/60 p-3 dark:border-slate-700 dark:bg-slate-800/40">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div className={`mt-1 font-mono text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}
