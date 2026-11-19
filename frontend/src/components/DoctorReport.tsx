/**
 * Doctor-ready consultation report.
 *
 * The existing summary is written for the patient. This one is written for
 * the five minutes a clinician actually has: what was reported, over what
 * period, what the tool concluded and with how much confidence, what safety
 * checks ran, and what the patient should ask.
 *
 * Composed client-side from data the app already holds -- the session
 * report, the local symptom diary, the local profile, and the condition
 * fact sheet -- and printed through the browser, so it becomes a PDF with
 * no server round-trip and no PDF dependency.
 *
 * Two things are deliberate. The differential is shown with probabilities
 * AND the abstention flag, so a clinician is not handed a number the tool
 * itself did not trust. And the red-flag section lists what was checked,
 * not merely that checking happened -- "no red flags" is only meaningful
 * if you can see which flags were looked for.
 */

import { useEffect, useState } from 'react';
import { api, conditionApi, ReportResponse, ConditionFact } from '../api/client';
import { useDiary, SEVERITY_LABELS } from '../hooks/useDiary';
import { formatProbability } from './viz/probability';
import { useProfile, AGE_LABELS, AgeGroup } from '../hooks/useProfile';

interface SafetyCategory { key: string; description: string; }

function fmtDate(iso?: string) {
  const d = iso ? new Date(iso) : new Date();
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'long', year: 'numeric' });
}

export default function DoctorReport({
  sessionId,
  onClose,
}: {
  sessionId: string;
  onClose: () => void;
}) {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [fact, setFact] = useState<ConditionFact | null>(null);
  const [checks, setChecks] = useState<SafetyCategory[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { entries, symptoms: diarySymptoms, seriesFor, trendFor } = useDiary();
  const { profile, isSet } = useProfile();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await api.report(sessionId);
        if (!alive) return;
        setReport(r);

        // Safety checks and the condition fact sheet are enrichment: if
        // either fails the report is still complete and printable.
        api.safetyCategories()
          .then((s) => alive && setChecks(s.categories))
          .catch(() => undefined);

        const top = (r.prediction as Record<string, unknown> | undefined)?.top_condition;
        if (typeof top === 'string') {
          conditionApi.match(top)
            .then((m) => { if (alive && m.status === 'matched' && m.condition) setFact(m.condition); })
            .catch(() => undefined);
        }
      } catch {
        if (alive) setError('Could not load the consultation. Please try again.');
      }
    })();
    return () => { alive = false; };
  }, [sessionId]);

  const pred = (report?.prediction ?? {}) as Record<string, any>;
  const alts: { disease: string; probability: number }[] = pred.alternatives ?? [];
  const abstained = Boolean(pred.abstained);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm print:static print:block print:bg-transparent print:p-0"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-label="Doctor-ready report"
        className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl dark:bg-slate-900 print:max-h-none print:rounded-none print:shadow-none"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-gray-200 px-5 py-3 dark:border-slate-700 print:hidden">
          <h2 className="text-base font-bold text-gray-800 dark:text-white">
            Report for your doctor
          </h2>
          <div className="flex gap-2">
            <button
              onClick={() => window.print()}
              disabled={!report}
              className="rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105 disabled:opacity-50"
            >
              Print / Save as PDF
            </button>
            <button
              onClick={onClose}
              className="rounded-xl px-3 py-2 text-sm text-gray-500 transition hover:bg-gray-100 dark:hover:bg-slate-700"
            >
              Close
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto bg-white px-8 py-6 text-[13px] leading-relaxed text-gray-900 print:overflow-visible print:px-0 print:py-0">
          {error && <p className="text-red-600">{error}</p>}
          {!report && !error && <p className="text-gray-400">Preparing the report…</p>}

          {report && (
            <article className="space-y-5">
              {/* ---- heading ---- */}
              <header className="border-b-2 border-gray-800 pb-2">
                <h1 className="text-xl font-extrabold tracking-tight">
                  Consultation summary
                </h1>
                <p className="text-[12px] text-gray-600">
                  Prepared by Arogya AI, an educational symptom-checking tool ·{' '}
                  {fmtDate(report.generated_at)} · {report.duration_turns} conversation turns
                </p>
              </header>

              <p className="rounded border-l-4 border-gray-800 bg-gray-50 px-3 py-2 text-[12px]">
                <strong>This is not a diagnosis.</strong> It was produced by an
                automated tool from a patient's own description of their symptoms,
                using a model trained on synthetic data. It is provided to support
                a clinical conversation, not to replace one.
              </p>

              {/* ---- patient ---- */}
              {isSet && (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Patient
                  </h2>
                  <dl className="grid grid-cols-2 gap-x-6 gap-y-0.5">
                    {profile.age_group && (
                      <div><dt className="inline font-semibold">Age band: </dt>
                        <dd className="inline">{AGE_LABELS[profile.age_group as AgeGroup]}</dd></div>
                    )}
                    {profile.sex && (
                      <div><dt className="inline font-semibold">Sex: </dt>
                        <dd className="inline">{profile.sex.replace(/_/g, ' ')}</dd></div>
                    )}
                    {profile.pregnant && (
                      <div><dt className="inline font-semibold">Pregnant: </dt>
                        <dd className="inline">yes</dd></div>
                    )}
                    {profile.known_conditions.length > 0 && (
                      <div className="col-span-2"><dt className="inline font-semibold">Known conditions: </dt>
                        <dd className="inline">{profile.known_conditions.join(', ')}</dd></div>
                    )}
                    {profile.allergies.length > 0 && (
                      <div className="col-span-2"><dt className="inline font-semibold">Allergies: </dt>
                        <dd className="inline">{profile.allergies.join(', ')}</dd></div>
                    )}
                  </dl>
                </section>
              )}

              {/* ---- symptoms ---- */}
              <section>
                <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                  Reported symptoms
                </h2>
                {report.symptoms.length ? (
                  <p>{report.symptoms.map((s) => s.replace(/_/g, ' ')).join(' · ')}</p>
                ) : (
                  <p className="italic text-gray-500">None recorded.</p>
                )}
              </section>

              {/* ---- consultation answers ---- */}
              {Object.keys(report.clinical_context ?? {}).length > 0 && (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Structured intake
                  </h2>
                  <table className="w-full">
                    <tbody>
                      {Object.entries(report.clinical_context).map(([k, v]) => (
                        <tr key={k} className="border-b border-gray-100 last:border-0">
                          <td className="py-0.5 pr-4 font-semibold">{k}</td>
                          <td className="py-0.5">{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              )}

              {/* ---- timeline from the diary ---- */}
              {diarySymptoms.length > 0 && (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Symptom timeline (patient-recorded)
                  </h2>
                  <table className="w-full">
                    <tbody>
                      {diarySymptoms.map((s) => {
                        const series = seriesFor(s);
                        const trend = trendFor(s);
                        return (
                          <tr key={s} className="border-b border-gray-100 last:border-0 align-top">
                            <td className="w-40 py-0.5 pr-4 font-semibold">{s}</td>
                            <td className="py-0.5">
                              {series.map((e) => `${e.date.slice(5)} ${SEVERITY_LABELS[e.severity]}`).join('  →  ')}
                              {trend && <em className="ml-2 text-gray-600">({trend})</em>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p className="mt-1 text-[11px] text-gray-500">
                    {entries.length} reading{entries.length === 1 ? '' : 's'} recorded by the patient
                    on their own device.
                  </p>
                </section>
              )}

              {/* ---- assessment ---- */}
              {report.prediction && (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Tool output
                  </h2>

                  {abstained ? (
                    <p className="mb-1.5 rounded bg-amber-50 px-2 py-1.5">
                      <strong>The tool declined to name a single condition.</strong>{' '}
                      {pred.abstain_reason === 'too_few_symptoms'
                        ? 'Too few symptoms were reported to support one.'
                        : 'No candidate was clearly ahead at this level of evidence.'}{' '}
                      The ranking below is shown for completeness only.
                    </p>
                  ) : (
                    <p className="mb-1.5">
                      <strong>Leading candidate:</strong> {pred.top_condition}{' '}
                      ({formatProbability(pred.confidence ?? 0)} calibrated probability)
                      {pred.severity && <> · baseline severity {pred.severity}</>}
                    </p>
                  )}

                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-300 text-left">
                        <th className="py-0.5 font-semibold">Differential</th>
                        <th className="w-24 py-0.5 text-right font-semibold">Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-gray-100">
                        <td className="py-0.5">{pred.top_condition}</td>
                        <td className="py-0.5 text-right tabular-nums">
                          {formatProbability(pred.confidence ?? 0)}
                        </td>
                      </tr>
                      {alts.map((a) => (
                        <tr key={a.disease} className="border-b border-gray-100 last:border-0">
                          <td className="py-0.5">{a.disease}</td>
                          <td className="py-0.5 text-right tabular-nums">
                            {formatProbability(a.probability)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {pred.triage && (
                    <p className="mt-1.5">
                      <strong>Suggested next step:</strong> {pred.triage.label} — {pred.triage.detail}
                    </p>
                  )}

                  <p className="mt-1.5 text-[11px] text-gray-600">
                    Probabilities come from a Bernoulli Naive Bayes classifier over
                    102 binary symptom features and 41 conditions, calibrated by
                    temperature scaling conditioned on how many symptoms were known.
                    They describe fit to a symptom pattern, not diagnostic certainty.
                  </p>
                </section>
              )}

              {/* ---- safety checks ---- */}
              {checks.length > 0 && (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Emergency checks performed
                  </h2>
                  <p className="text-[12px]">
                    Every message was screened against these {checks.length} categories
                    before any assessment was produced
                    {report.prediction ? ', and none matched' : ''}:
                  </p>
                  <p className="mt-0.5 text-[12px] text-gray-700">
                    {checks.map((c) => c.description).join(' · ')}
                  </p>
                </section>
              )}

              {/* ---- questions ---- */}
              {fact?.questions_to_ask_doctor?.length ? (
                <section>
                  <h2 className="mb-1 border-b border-gray-300 text-[11px] font-bold uppercase tracking-wider">
                    Questions the patient may want to ask
                  </h2>
                  <ol className="list-decimal space-y-0.5 pl-5">
                    {fact.questions_to_ask_doctor.map((q) => <li key={q}>{q}</li>)}
                  </ol>
                </section>
              ) : null}

              <footer className="border-t border-gray-300 pt-2 text-[11px] text-gray-500">
                Generated by Arogya AI · session {report.session_id.slice(0, 8)} ·
                educational use only · not a medical diagnosis.
              </footer>
            </article>
          )}
        </div>
      </div>
    </div>
  );
}
