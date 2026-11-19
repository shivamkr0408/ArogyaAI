/**
 * Past consultations, as a health timeline.
 *
 * Two jobs. First, a way back: "New chat" used to make the previous
 * consultation unreachable, and this restores it. Second, the longitudinal
 * view -- seeing the same complaint three times in a fortnight, each time
 * triaged a little higher, is information no single consultation contains.
 *
 * The list is an index held on the device. Opening a row re-fetches the
 * transcript from the backend, so nothing clinical is duplicated into
 * localStorage.
 */

import { useState } from 'react';
import { useChat } from '../hooks/useChatSession';
import type { ConsultationRecord } from '../hooks/useHistory';

const TRIAGE_STYLE: Record<string, { dot: string; text: string }> = {
  self_care: { dot: 'bg-emerald-500', text: 'text-emerald-700 dark:text-emerald-300' },
  soon: { dot: 'bg-amber-500', text: 'text-amber-700 dark:text-amber-300' },
  urgent: { dot: 'bg-red-500', text: 'text-red-700 dark:text-red-300' },
};

function relativeDay(iso: string): string {
  const then = new Date(iso);
  const today = new Date();
  const days = Math.floor(
    (new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime() -
      new Date(then.getFullYear(), then.getMonth(), then.getDate()).getTime()) / 86400000,
  );
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return then.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

function Row({ rec, active, onOpen, onRemove }: {
  rec: ConsultationRecord;
  active: boolean;
  onOpen: () => void;
  onRemove: () => void;
}) {
  const triage = rec.triageLevel ? TRIAGE_STYLE[rec.triageLevel] : undefined;
  return (
    <li
      className={`group relative rounded-xl border px-3 py-2.5 transition ${
        active
          ? 'border-brand-300 bg-brand-50/70 dark:border-brand-700 dark:bg-brand-900/20'
          : 'border-gray-200 hover:border-brand-200 hover:bg-brand-50/40 dark:border-slate-700 dark:hover:bg-slate-800/60'
      }`}
    >
      <button onClick={onOpen} className="w-full text-left">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[11px] font-medium text-gray-400 dark:text-gray-500">
            {relativeDay(rec.updatedAt)} · {rec.turns} turn{rec.turns === 1 ? '' : 's'}
          </span>
          {active && (
            <span className="text-[10px] font-semibold text-brand-600 dark:text-brand-300">
              open now
            </span>
          )}
        </div>

        <p className="mt-0.5 line-clamp-2 text-sm text-gray-700 dark:text-gray-200">
          {rec.opening}
        </p>

        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          {rec.emergency ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-red-50 px-2 py-0.5 text-[11px] font-semibold text-red-700 dark:bg-red-900/30 dark:text-red-300">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              Emergency advised
            </span>
          ) : rec.condition ? (
            <>
              <span className="text-[11px] font-semibold text-gray-700 dark:text-gray-200">
                {rec.condition}
              </span>
              {rec.confidence !== undefined && (
                <span className="text-[11px] tabular-nums text-gray-400 dark:text-gray-500">
                  {Math.round(rec.confidence * 100)}%
                </span>
              )}
            </>
          ) : (
            <span className="text-[11px] italic text-gray-400 dark:text-gray-500">
              no result reached
            </span>
          )}

          {triage && rec.triageLabel && (
            <span className={`inline-flex items-center gap-1.5 text-[11px] ${triage.text}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${triage.dot}`} />
              {rec.triageLabel}
            </span>
          )}
        </div>
      </button>

      <button
        onClick={onRemove}
        title="Remove from history"
        aria-label="Remove from history"
        className="absolute right-2 top-2 opacity-0 transition group-hover:opacity-100 focus:opacity-100 text-gray-300 hover:text-red-500 dark:text-slate-600"
      >
        ✕
      </button>
    </li>
  );
}

export default function HistoryPanel() {
  const [open, setOpen] = useState(false);
  const { history, sessionId, openSession, removeFromHistory, clearHistory } = useChat();

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="Your past consultations on this device"
        className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-gray-600 transition hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-slate-700 sm:px-3"
      >
        🕑 <span className="hidden md:inline">History</span>
        {history.length > 0 && (
          <span className="ml-1 rounded-full bg-brand-100 px-1.5 text-[10px] font-semibold text-brand-700 dark:bg-brand-900/50 dark:text-brand-300">
            {history.length}
          </span>
        )}
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex justify-end bg-black/30 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <aside
            className="flex h-full w-full max-w-md flex-col bg-white shadow-2xl animate-slide-in-right dark:bg-slate-900"
            onClick={e => e.stopPropagation()}
          >
            <header className="flex items-center justify-between border-b border-gray-200 px-5 py-4 dark:border-slate-700">
              <div>
                <h2 className="text-base font-bold text-gray-800 dark:text-white">
                  Your consultations
                </h2>
                <p className="text-[11px] text-gray-500 dark:text-gray-400">
                  Kept on this device only — never uploaded
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                aria-label="Close history"
                className="rounded-lg px-2 py-1 text-gray-400 transition hover:bg-gray-100 dark:hover:bg-slate-700"
              >
                ✕
              </button>
            </header>

            <div className="flex-1 overflow-y-auto px-4 py-4">
              {history.length === 0 ? (
                <div className="mt-16 text-center">
                  <p className="text-4xl">🕑</p>
                  <p className="mt-3 text-sm font-medium text-gray-600 dark:text-gray-300">
                    No past consultations yet
                  </p>
                  <p className="mx-auto mt-1 max-w-xs text-xs leading-relaxed text-gray-400 dark:text-gray-500">
                    Once you describe your symptoms, each consultation is listed
                    here so you can reopen it or compare how things changed.
                  </p>
                </div>
              ) : (
                <ul className="space-y-2">
                  {history.map(rec => (
                    <Row
                      key={rec.id}
                      rec={rec}
                      active={rec.id === sessionId}
                      onOpen={async () => { await openSession(rec.id); setOpen(false); }}
                      onRemove={() => removeFromHistory(rec.id)}
                    />
                  ))}
                </ul>
              )}
            </div>

            {history.length > 0 && (
              <footer className="border-t border-gray-200 px-5 py-3 dark:border-slate-700">
                <button
                  onClick={() => { if (confirm('Remove all saved consultations from this device?')) clearHistory(); }}
                  className="text-xs font-medium text-gray-400 transition hover:text-red-500"
                >
                  Clear all history
                </button>
              </footer>
            )}
          </aside>
        </div>
      )}
    </>
  );
}
