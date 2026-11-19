import { useEffect, useState } from 'react';
import { api, ReportResponse } from '../api/client';

interface Props {
  sessionId: string;
  onClose: () => void;
}

/**
 * Consultation summary the user can print or save and take to a doctor.
 * Rendered from the backend-built markdown so the file and the on-screen
 * view can never drift apart.
 */
export default function ReportModal({ sessionId, onClose }: Props) {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .report(sessionId)
      .then(setReport)
      .catch(() => setError('Could not load the consultation summary.'));
  }, [sessionId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const download = () => {
    if (!report) return;
    const blob = new Blob([report.markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `arogya-consultation-${sessionId.slice(0, 8)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copy = () => {
    if (report) navigator.clipboard?.writeText(report.markdown);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in-up"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-800 rounded-2xl shadow-soft w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-5 py-3.5 border-b border-gray-100 dark:border-slate-700">
          <div>
            <h2 className="font-bold text-gray-800 dark:text-white">Consultation summary</h2>
            <p className="text-[11px] text-gray-400 dark:text-gray-500">
              Save this and share it with a healthcare professional
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-8 h-8 rounded-lg text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-700 transition"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
          {!report && !error && (
            <p className="text-sm text-gray-400 dark:text-gray-500">Building summary…</p>
          )}
          {report && (
            <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-gray-700 dark:text-gray-300">
              {report.markdown}
            </pre>
          )}
        </div>

        <footer className="flex flex-wrap items-center justify-end gap-2 px-5 py-3 border-t border-gray-100 dark:border-slate-700 bg-gray-50/60 dark:bg-slate-900/40">
          <button
            onClick={copy}
            disabled={!report}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-gray-300 hover:bg-white dark:hover:bg-slate-700 disabled:opacity-40 transition"
          >
            Copy
          </button>
          <button
            onClick={() => window.print()}
            disabled={!report}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-gray-300 hover:bg-white dark:hover:bg-slate-700 disabled:opacity-40 transition"
          >
            Print
          </button>
          <button
            onClick={download}
            disabled={!report}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-brand-500 to-indigo-600 text-white hover:brightness-105 disabled:opacity-40 transition"
          >
            Download .md
          </button>
        </footer>
      </div>
    </div>
  );
}
