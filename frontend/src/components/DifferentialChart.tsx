import { formatProbability } from './viz/probability';

interface Row {
  disease: string;
  probability: number;
}

interface Props {
  top: Row;
  alternatives: Row[];
}

/**
 * Ranked differential: the job is magnitude (how likely, relative to each
 * other), not identity — so this is a single-hue horizontal bar set rather
 * than a categorical palette. The leading condition carries the full brand
 * hue; the rest step down the same ramp.
 */
export default function DifferentialChart({ top, alternatives }: Props) {
  const rows = [top, ...alternatives];
  const max = Math.max(...rows.map((r) => r.probability), 0.01);

  return (
    <figure className="mt-4 m-0">
      <figcaption className="text-xs uppercase tracking-wider text-gray-400 dark:text-gray-500 font-semibold mb-2.5">
        Differential — how the model ranked the candidates
      </figcaption>

      <div className="space-y-2">
        {rows.map((row, i) => {
          const pct = formatProbability(row.probability);
          const width = Math.max((row.probability / max) * 100, 2);
          const leading = i === 0;
          return (
            <div
              key={row.disease}
              className="group grid grid-cols-[minmax(0,9rem)_1fr_2.5rem] items-center gap-2.5"
              title={`${row.disease}: ${pct} posterior probability`}
            >
              <span
                className={`text-xs truncate ${
                  leading
                    ? 'font-semibold text-gray-800 dark:text-gray-100'
                    : 'text-gray-500 dark:text-gray-400'
                }`}
              >
                {row.disease}
              </span>

              <div className="h-2.5 rounded-full bg-gray-100 dark:bg-slate-700/70 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${width}%`,
                    backgroundColor: leading
                      ? 'rgb(46 125 111)'
                      : i === 1
                        ? 'rgb(92 178 160)'
                        : i === 2
                          ? 'rgb(137 205 189)'
                          : 'rgb(181 226 215)',
                  }}
                />
              </div>

              <span
                className={`text-xs tabular-nums text-right ${
                  leading
                    ? 'font-semibold text-gray-700 dark:text-gray-200'
                    : 'text-gray-400 dark:text-gray-500'
                }`}
              >
                {pct}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-[10px] text-gray-400 dark:text-gray-600 mt-2">
        Posterior probabilities from the Naive Bayes classifier. They describe fit to the
        symptom pattern, not certainty of a diagnosis.
      </p>
    </figure>
  );
}
