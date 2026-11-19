/**
 * Formatting for displayed probabilities.
 *
 * Math.round() was turning 0.999 into "100%" and 0.004 into "0%". Both are
 * lies of presentation: a classifier is never certain, and a candidate it
 * still ranks is not impossible. On the doctor-facing report especially, a
 * "100%" would undermine the whole document.
 *
 * So the display saturates rather than rounding through the endpoints.
 */
export function formatProbability(p: number): string {
  const pct = p * 100;
  if (pct >= 99.5) return '>99%';
  if (pct > 0 && pct < 0.5) return '<1%';
  if (pct >= 10) return `${Math.round(pct)}%`;
  return `${pct.toFixed(pct < 1 ? 1 : 0)}%`;
}
