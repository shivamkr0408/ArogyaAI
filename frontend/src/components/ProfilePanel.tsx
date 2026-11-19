/**
 * Health profile editor, plus the "clear my data" control.
 *
 * The profile exists to make the safety layer person-aware, so the panel
 * says what each field is actually used for rather than collecting it
 * because a form usually does. Nothing here leaves the device except as
 * part of a turn, and the clear button removes every key this app writes.
 */

import { useState } from 'react';
import {
  useProfile,
  AGE_LABELS,
  AgeGroup,
  HealthProfile,
} from '../hooks/useProfile';

const CONDITIONS = [
  'Diabetes', 'Hypertension', 'Asthma', 'COPD', 'Heart disease',
  'Kidney disease', 'Liver disease', 'Immunosuppressed',
];

function Chip({ on, children, onClick }: {
  on: boolean; children: React.ReactNode; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={on}
      className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
        on
          ? 'border-brand-500 bg-brand-500 text-white'
          : 'border-gray-200 text-gray-600 hover:border-brand-300 dark:border-slate-600 dark:text-gray-300'
      }`}
    >
      {on && '✓ '}{children}
    </button>
  );
}

export default function ProfilePanel({ onClose }: { onClose: () => void }) {
  const { profile, update, clearAll } = useProfile();
  const [draft, setDraft] = useState<HealthProfile>(profile);
  const [cleared, setCleared] = useState(false);

  const toggle = (list: string[], value: string) =>
    list.includes(value) ? list.filter((v) => v !== value) : [...list, value];

  const saveAndClose = () => { update(draft); onClose(); };

  // Pregnancy only applies to some profiles, and asking everyone is both
  // noisy and a little insulting.
  const showPregnancy = draft.sex === 'female' &&
    (draft.age_group === '13_17' || draft.age_group === '18_64');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-label="Health profile"
        className="flex max-h-[88vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl animate-fade-in-up dark:bg-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between border-b border-gray-200 px-5 py-4 dark:border-slate-700">
          <div>
            <h2 className="text-base font-bold text-gray-800 dark:text-white">Health profile</h2>
            <p className="mt-0.5 text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
              Used to make the safety checks person-aware. Kept on this device
              and sent only as part of a question — never stored on a server.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close profile"
            className="rounded-lg px-2 py-1 text-gray-400 transition hover:bg-gray-100 dark:hover:bg-slate-700"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Age group
            </h3>
            <p className="mb-2 mt-0.5 text-[11px] text-gray-400 dark:text-gray-500">
              Changes how fever and breathing symptoms are treated at the extremes of age.
            </p>
            <div className="flex flex-wrap gap-2">
              {(Object.keys(AGE_LABELS) as AgeGroup[]).map((k) => (
                <Chip key={k} on={draft.age_group === k}
                      onClick={() => setDraft({ ...draft, age_group: draft.age_group === k ? undefined : k })}>
                  {AGE_LABELS[k]}
                </Chip>
              ))}
            </div>
          </section>

          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Sex
            </h3>
            <div className="mt-2 flex flex-wrap gap-2">
              {(['female', 'male', 'other', 'prefer_not_to_say'] as const).map((k) => (
                <Chip key={k} on={draft.sex === k}
                      onClick={() => setDraft({ ...draft, sex: draft.sex === k ? undefined : k })}>
                  {k === 'prefer_not_to_say' ? 'Prefer not to say' : k[0].toUpperCase() + k.slice(1)}
                </Chip>
              ))}
            </div>
          </section>

          {showPregnancy && (
            <section>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Pregnancy
              </h3>
              <p className="mb-2 mt-0.5 text-[11px] text-gray-400 dark:text-gray-500">
                Several symptoms are assessed more urgently during pregnancy.
              </p>
              <Chip on={draft.pregnant} onClick={() => setDraft({ ...draft, pregnant: !draft.pregnant })}>
                I am pregnant
              </Chip>
            </section>
          )}

          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Known conditions
            </h3>
            <div className="mt-2 flex flex-wrap gap-2">
              {CONDITIONS.map((c) => (
                <Chip key={c} on={draft.known_conditions.includes(c)}
                      onClick={() => setDraft({ ...draft, known_conditions: toggle(draft.known_conditions, c) })}>
                  {c}
                </Chip>
              ))}
            </div>
          </section>

          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Allergies
            </h3>
            <input
              value={draft.allergies.join(', ')}
              onChange={(e) => setDraft({
                ...draft,
                allergies: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
              })}
              placeholder="e.g. penicillin, peanuts"
              aria-label="Allergies, comma separated"
              className="mt-2 w-full rounded-xl border border-gray-200 bg-transparent px-3 py-2 text-sm text-gray-800 placeholder-gray-400 focus:border-brand-400 focus:outline-none dark:border-slate-600 dark:text-gray-100"
            />
          </section>

          <section className="rounded-xl border border-red-200 bg-red-50/60 p-3 dark:border-red-900 dark:bg-red-900/20">
            <h3 className="text-xs font-semibold text-red-800 dark:text-red-300">Your data</h3>
            <p className="mt-0.5 text-[11px] leading-relaxed text-red-700/90 dark:text-red-300/80">
              Removes your profile, your saved consultations and your symptom
              diary from this device. It cannot be undone.
            </p>
            <button
              onClick={() => {
                if (confirm('Remove your profile, consultation history and symptom diary from this device?')) {
                  clearAll();
                  setDraft({ pregnant: false, known_conditions: [], allergies: [] });
                  setCleared(true);
                }
              }}
              className="mt-2 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-xs font-semibold text-red-700 transition hover:bg-red-50 dark:border-red-800 dark:bg-slate-800 dark:text-red-300"
            >
              Clear my data
            </button>
            {cleared && (
              <p className="mt-1.5 text-[11px] font-medium text-red-700 dark:text-red-300">
                Cleared from this device.
              </p>
            )}
          </section>
        </div>

        <footer className="flex justify-end gap-2 border-t border-gray-200 px-5 py-3 dark:border-slate-700">
          <button onClick={onClose}
                  className="rounded-xl px-3 py-2 text-sm text-gray-500 transition hover:bg-gray-100 dark:hover:bg-slate-700">
            Cancel
          </button>
          <button onClick={saveAndClose}
                  className="rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105">
            Save
          </button>
        </footer>
      </div>
    </div>
  );
}
