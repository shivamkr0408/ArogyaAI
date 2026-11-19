/**
 * Chart colour roles.
 *
 * These are the validated reference values from the project's data-viz
 * palette, checked with the palette validator in both light and dark mode:
 *
 *   diverging poles  #2a78d6 / #e34948 (light), #3987e5 / #e66767 (dark)
 *   -> all six checks PASS in both modes
 *
 * Charts here use three colour jobs and no categorical palette, which is
 * deliberate:
 *
 *  - **sequential** (one hue, more-is-darker) for plain magnitude, e.g. the
 *    differential probabilities;
 *  - **emphasis** (one accent + de-emphasis grey) for the model comparison,
 *    because the story is "this is the one we deployed", not "tell eight
 *    models apart";
 *  - **diverging** (blue <-> red around a neutral grey midpoint) for the
 *    Naive Bayes log-odds, which are genuinely signed - evidence for versus
 *    against a condition.
 *
 * The de-emphasis grey is intentionally below the chroma floor: it is
 * recessive context, not a series identity, so the categorical gate does not
 * apply to it.
 */

export const VIZ = {
  /** Sequential blue ramp, light mode. */
  seqLight: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#184f95'],
  /** Diverging poles + neutral midpoint. */
  positiveLight: '#2a78d6',
  negativeLight: '#e34948',
  midLight: '#f0efec',
  positiveDark: '#3987e5',
  negativeDark: '#e66767',
  midDark: '#383835',
} as const;

/** Tailwind classes for the two diverging poles, so dark mode swaps cleanly. */
export const POLE_POSITIVE = 'bg-[#2a78d6] dark:bg-[#3987e5]';
export const POLE_NEGATIVE = 'bg-[#e34948] dark:bg-[#e66767]';
export const POLE_NEUTRAL = 'bg-[#f0efec] dark:bg-[#383835]';

/** Emphasis pair: accent for the subject, grey for context. */
export const EMPHASIS_ON = 'bg-[#2a78d6] dark:bg-[#3987e5]';
export const EMPHASIS_OFF = 'bg-slate-300 dark:bg-slate-600';

/** Format a 0-1 metric as a percentage with one decimal. */
export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Format a raw 0-1 score for a dense table. */
export function dec(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return value.toFixed(digits);
}
