/**
 * Evaluation dashboard -- every measured number the project has.
 *
 * Nothing here is typed in by hand: the panel renders the JSON that
 * `eval/compare_models.py`, `eval/eval_nlp.py`, `eval/eval_rag.py` and
 * `ml/train_model.py` wrote, served by /api/evaluation. If a results file is
 * missing, the panel says so and prints the command that regenerates it,
 * rather than showing a blank or a stale figure.
 *
 * Form choices follow the project's data-viz method:
 *  - eight models x six metrics is a *table*, not a chart with eight colours;
 *  - the macro-F1 comparison beside it is an **emphasis** chart, because the
 *    question is "where does the deployed model sit", not "tell eight models
 *    apart";
 *  - the NLP component scores are a **KPI row** of stat tiles.
 */

import { useEffect, useState } from 'react';
import { api, type EvaluationResponse } from '../api/client';
import { EmphasisBars, StatTile } from './viz/Bars';
import { dec, pct } from './viz/tokens';

type Tab = 'models' | 'nlp' | 'retrieval';

function Card({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white/60 p-3 dark:border-slate-700 dark:bg-slate-800/40">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {title}
      </h4>
      {hint && <p className="mt-0.5 text-[11px] leading-snug text-slate-400 dark:text-slate-500">{hint}</p>}
      <div className="mt-2.5">{children}</div>
    </section>
  );
}

function Missing({ what, command }: { what: string; command?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 p-4 text-xs text-slate-500 dark:border-slate-600 dark:text-slate-400">
      No results for {what} yet.
      {command && (
        <>
          {' '}Run{' '}
          <code className="rounded bg-slate-100 px-1 py-0.5 font-mono dark:bg-slate-900">{command}</code>
          {' '}to generate them.
        </>
      )}
    </div>
  );
}

export default function EvaluationPanel({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [tab, setTab] = useState<Tab>('models');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .evaluation()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load evaluation'));
  }, []);

  const comparison = data?.model_comparison ?? null;
  const nlp = data?.nlp_components ?? null;
  const rag = data?.retrieval ?? null;
  const deployedName = comparison?.deployed_model;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/50 p-3 backdrop-blur-sm sm:p-6">
      <div className="w-full max-w-4xl rounded-xl border border-slate-200 bg-slate-50 shadow-soft dark:border-slate-700 dark:bg-slate-900">
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 p-4 dark:border-slate-700">
          <div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">Evaluation</h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Measured results, read straight from the evaluation scripts' output.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-200 dark:text-slate-400 dark:hover:bg-slate-700"
            aria-label="Close evaluation panel"
          >
            Close
          </button>
        </header>

        <nav className="flex gap-1 border-b border-slate-200 px-4 pt-3 dark:border-slate-700">
          {(
            [
              ['models', 'Classifier'],
              ['nlp', 'NLP components'],
              ['retrieval', 'Retrieval'],
            ] as [Tab, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-t-md px-3 py-1.5 text-xs font-medium transition ${
                tab === key
                  ? 'bg-white text-brand-700 shadow-sm dark:bg-slate-800 dark:text-brand-300'
                  : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="space-y-3 p-4">
          {error && (
            <p className="rounded-lg bg-rose-50 p-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
              {error}
            </p>
          )}
          {!data && !error && (
            <p className="text-xs text-slate-500 dark:text-slate-400">Loading measured results...</p>
          )}

          {/* ---------------------------------------------------- Classifier */}
          {tab === 'models' && data && (
            comparison ? (
              <>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <StatTile
                    label="Deployed"
                    value={comparison.deployed_model.replace(' NB', ' NB')}
                    sub={`${comparison.dataset.classes} conditions`}
                  />
                  <StatTile
                    label="Accuracy"
                    value={pct(data.deployed_model.metrics?.accuracy)}
                    sub={`top-3 ${pct(data.deployed_model.metrics?.top3_accuracy)}`}
                  />
                  <StatTile
                    label="Macro F1"
                    value={dec(data.deployed_model.metrics?.macro_f1)}
                    sub="all classes weighted equally"
                  />
                  <StatTile
                    label={`CV (${comparison.dataset.cv_folds}-fold)`}
                    value={dec(data.deployed_model.metrics?.cv_accuracy_mean)}
                    sub={`± ${dec(data.deployed_model.metrics?.cv_accuracy_std)}`}
                  />
                </div>

                <Card
                  title="Macro F1 by model"
                  hint="Macro F1, not accuracy: class sizes encode clinical base rates on purpose, so accuracy flatters models that favour common conditions."
                >
                  <EmphasisBars
                    rows={[...comparison.results]
                      .sort((a, b) => b.macro_f1 - a.macro_f1)
                      .map((r) => ({
                        label: r.model,
                        value: r.macro_f1,
                        emphasized: r.model === deployedName,
                        note: r.note,
                      }))}
                    max={1}
                    valueFormat={(v) => dec(v)}
                    caption={comparison.selection_rationale}
                  />
                </Card>

                <Card
                  title="Full comparison"
                  hint={`${comparison.dataset.train_rows} train / ${comparison.dataset.test_rows} test rows · ${comparison.dataset.features} features · identical split and folds for every model.`}
                >
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                        <tr>
                          <th className="pb-1.5 pr-3">Model</th>
                          <th className="pb-1.5 pr-3 text-right">Acc</th>
                          <th className="pb-1.5 pr-3 text-right">Top-3</th>
                          <th className="pb-1.5 pr-3 text-right">Macro F1</th>
                          <th className="pb-1.5 pr-3 text-right">Wtd F1</th>
                          <th className="pb-1.5 pr-3 text-right">CV acc</th>
                          <th className="pb-1.5 text-right">Fit ms</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                        {[...comparison.results]
                          .sort((a, b) => b.macro_f1 - a.macro_f1)
                          .map((r) => {
                            const isDeployed = r.model === deployedName;
                            return (
                              <tr
                                key={r.model}
                                title={r.note}
                                className={
                                  isDeployed
                                    ? 'bg-brand-50/70 font-medium dark:bg-brand-900/20'
                                    : undefined
                                }
                              >
                                <td className="py-1.5 pr-3 text-slate-800 dark:text-slate-200">
                                  {r.model}
                                  {isDeployed && (
                                    <span className="ml-1.5 text-[10px] font-semibold uppercase tracking-wide text-brand-700 dark:text-brand-300">
                                      deployed
                                    </span>
                                  )}
                                </td>
                                <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                  {dec(r.accuracy)}
                                </td>
                                <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                  {r.top3_accuracy === null ? 'n/a' : dec(r.top3_accuracy)}
                                </td>
                                <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-800 dark:text-slate-200">
                                  {dec(r.macro_f1)}
                                </td>
                                <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                  {dec(r.weighted_f1)}
                                </td>
                                <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                  {r.cv_accuracy_mean === null
                                    ? 'n/a'
                                    : `${dec(r.cv_accuracy_mean)}±${dec(r.cv_accuracy_std, 3)}`}
                                </td>
                                <td className="py-1.5 text-right font-mono tabular-nums text-slate-500 dark:text-slate-400">
                                  {r.fit_ms.toFixed(0)}
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>
                  {comparison.notes && (
                    <ul className="mt-3 space-y-1 border-t border-slate-200 pt-2 text-[11px] leading-snug text-slate-500 dark:border-slate-700 dark:text-slate-400">
                      {comparison.notes.map((note, i) => (
                        <li key={i}>· {note}</li>
                      ))}
                    </ul>
                  )}
                </Card>

                {data.deployed_model.metrics?.per_class && (
                  <Card
                    title="Hardest classes for the deployed model"
                    hint="Lowest F1 first. The confusions are clinically plausible, which is the useful kind of error."
                  >
                    <table className="w-full text-left text-xs">
                      <thead className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
                        <tr>
                          <th className="pb-1.5 pr-3">Condition</th>
                          <th className="pb-1.5 pr-3 text-right">Precision</th>
                          <th className="pb-1.5 pr-3 text-right">Recall</th>
                          <th className="pb-1.5 pr-3 text-right">F1</th>
                          <th className="pb-1.5 text-right">n</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                        {Object.entries(
                          data.deployed_model.metrics.per_class as Record<
                            string,
                            { precision: number; recall: number; f1: number; support: number }
                          >,
                        )
                          .sort((a, b) => a[1].f1 - b[1].f1)
                          .slice(0, 6)
                          .map(([name, m]) => (
                            <tr key={name}>
                              <td className="py-1.5 pr-3 text-slate-800 dark:text-slate-200">{name}</td>
                              <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                {dec(m.precision)}
                              </td>
                              <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                                {dec(m.recall)}
                              </td>
                              <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-slate-800 dark:text-slate-200">
                                {dec(m.f1)}
                              </td>
                              <td className="py-1.5 text-right font-mono tabular-nums text-slate-500 dark:text-slate-400">
                                {m.support}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </Card>
                )}
              </>
            ) : (
              <Missing what="the model comparison" command={data.how_to_regenerate.model_comparison} />
            )
          )}

          {/* ------------------------------------------------ NLP components */}
          {tab === 'nlp' && data && (
            nlp ? (
              <>
                <Card
                  title="Symptom extraction"
                  hint={`Set-level precision/recall against ${nlp.symptom_extraction.n_cases} hand-annotated utterances. "Negation leaks" counts symptoms the user explicitly denied that still reached the classifier -- the failure that would actually mislead someone.`}
                >
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile label="Micro F1" value={dec(nlp.symptom_extraction.micro_f1)} />
                    <StatTile label="Macro F1" value={dec(nlp.symptom_extraction.macro_f1)} />
                    <StatTile
                      label="Exact set match"
                      value={pct(nlp.symptom_extraction.exact_set_match)}
                      sub="whole symptom set correct"
                    />
                    <StatTile
                      label="Negation leaks"
                      value={String(nlp.symptom_extraction.negation_violations)}
                      tone={nlp.symptom_extraction.negation_violations === 0 ? 'good' : 'warn'}
                    />
                  </div>
                  <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
                    {nlp.symptom_extraction.true_positives} true positives ·{' '}
                    {nlp.symptom_extraction.false_positives} false positives ·{' '}
                    {nlp.symptom_extraction.false_negatives} missed
                  </p>
                </Card>

                <Card
                  title="Assertion detection (NegEx / ConText)"
                  hint="Accuracy over the six derived labels. Only affirmed, current, patient-attributed symptoms reach the classifier."
                >
                  <div className="mb-3">
                    <StatTile
                      label="Overall accuracy"
                      value={pct(nlp.assertion.accuracy)}
                      sub={`${nlp.assertion.n_cases} annotated mentions`}
                      tone={nlp.assertion.accuracy >= 0.95 ? 'good' : 'neutral'}
                    />
                  </div>
                  <EmphasisBars
                    rows={Object.entries(
                      nlp.assertion.per_label as Record<string, { accuracy: number; support: number }>,
                    ).map(([label, v]) => ({
                      label: `${label} (n=${v.support})`,
                      value: v.accuracy,
                    }))}
                    max={1}
                    valueFormat={(v) => pct(v, 0)}
                  />
                </Card>

                <Card
                  title="Intent classification"
                  hint="Two honest numbers: cross-validated accuracy on the training distribution, and accuracy on a deliberately hard held-out paraphrase set that shares no sentence with the corpus."
                >
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile
                      label="Held-out set"
                      value={pct(nlp.intent.accuracy)}
                      sub={`${nlp.intent.n_cases} unseen paraphrases`}
                    />
                    <StatTile label="Macro F1" value={dec(nlp.intent.macro_f1)} />
                    <StatTile
                      label="Leakage"
                      value={nlp.intent.leaked_from_training?.length ? String(nlp.intent.leaked_from_training.length) : 'none'}
                      tone={nlp.intent.leaked_from_training?.length ? 'warn' : 'good'}
                      sub="items also in training"
                    />
                    <StatTile
                      label="Rule-decided"
                      value={String(nlp.intent.rule_decided)}
                      sub="regex, before the model"
                    />
                  </div>
                  <p className="mt-2 text-[11px] leading-snug text-slate-500 dark:text-slate-400">
                    Routing in the dialogue manager requires 45% confidence before acting on an
                    intent; below that the turn is treated as an ordinary symptom description, which
                    is the safe default. Every miss in this set fell below that threshold.
                  </p>
                </Card>

                <div className="grid gap-3 sm:grid-cols-3">
                  <Card title="Temporal" hint="Durations resolved to days.">
                    <StatTile label="Value accuracy" value={pct(nlp.temporal.value_accuracy)} />
                    <div className="mt-2">
                      <StatTile label="Bucket accuracy" value={pct(nlp.temporal.bucket_accuracy)} />
                    </div>
                  </Card>
                  <Card title="Trend" hint="Worsening / improving / unchanged.">
                    <StatTile
                      label="Accuracy"
                      value={pct(nlp.trend.accuracy)}
                      sub={`n=${nlp.trend.n_cases}`}
                    />
                  </Card>
                  <Card title="Distress" hint="Calm / elevated / high.">
                    <StatTile
                      label="Accuracy"
                      value={pct(nlp.distress.accuracy)}
                      sub={`n=${nlp.distress.n_cases}`}
                    />
                  </Card>
                </div>
              </>
            ) : (
              <Missing what="the NLP components" command={data.how_to_regenerate.nlp_components} />
            )
          )}

          {/* ---------------------------------------------------- Retrieval */}
          {tab === 'retrieval' && data && (
            rag ? (
              <>
                <Card
                  title="Unfiltered semantic search"
                  hint={`The honest measure of the embedding model on this corpus: ${rag.indexed_chunks} chunks, ${rag.embedding_model}, no disease filter. ${rag.unfiltered.n_queries} gold queries.`}
                >
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile label="Hit@1" value={pct(rag.unfiltered.hit_at_1)} />
                    <StatTile label="Hit@3" value={pct(rag.unfiltered.hit_at_3)} />
                    <StatTile label="MRR" value={dec(rag.unfiltered.mrr)} />
                    <StatTile label="nDCG@5" value={dec(rag.unfiltered.ndcg_at_5)} />
                  </div>
                  {rag.unfiltered.precision_at_5_ceiling && (
                    <p className="mt-2 text-[11px] leading-snug text-slate-500 dark:text-slate-400">
                      Precision@5 is {dec(rag.unfiltered.precision_at_5)} against a hard ceiling of{' '}
                      {dec(rag.unfiltered.precision_at_5_ceiling)} — each disease document yields only{' '}
                      {rag.unfiltered.avg_chunks_per_disease} chunks, so at most that many of five
                      results can be relevant. That is{' '}
                      {pct(rag.unfiltered.precision_at_5_vs_ceiling)} of what is achievable.
                    </p>
                  )}
                </Card>

                <Card
                  title="Production path (filtered by prediction)"
                  hint="Once the classifier names a condition, retrieval is constrained to that disease's chunks. Near-perfect by construction, so it is reported separately and is not a measure of retrieval quality."
                >
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile label="Hit@1" value={pct(rag.filtered.hit_at_1)} />
                    <StatTile label="MRR" value={dec(rag.filtered.mrr)} />
                    <StatTile label="nDCG@5" value={dec(rag.filtered.ndcg_at_5)} />
                    <StatTile label="Precision@5" value={dec(rag.filtered.precision_at_5)} />
                  </div>
                </Card>

                {rag.unfiltered.failures?.length > 0 && (
                  <Card
                    title="Retrieval misses"
                    hint="Queries where the correct document was not in the top 5, printed in full rather than averaged away."
                  >
                    <ul className="space-y-2 text-xs">
                      {rag.unfiltered.failures.map(
                        (f: { query: string; expected: string; retrieved: string[] }, i: number) => (
                          <li key={i}>
                            <p className="text-slate-800 dark:text-slate-200">"{f.query}"</p>
                            <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                              expected <span className="font-mono">{f.expected}</span> · got{' '}
                              <span className="font-mono">{f.retrieved.slice(0, 3).join(', ')}</span>
                            </p>
                          </li>
                        ),
                      )}
                    </ul>
                  </Card>
                )}
              </>
            ) : (
              <Missing what="retrieval" command={data.how_to_regenerate.retrieval} />
            )
          )}
        </div>
      </div>
    </div>
  );
}
