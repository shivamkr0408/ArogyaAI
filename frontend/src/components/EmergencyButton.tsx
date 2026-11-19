/**
 * Always-reachable emergency access.
 *
 * Every other safety feature in this app is reactive: it fires when the
 * red-flag layer recognises something. This one is for the case the model
 * misses, or the case where the person already knows it is serious and
 * should not have to describe symptoms to a chatbot first. It is therefore
 * present on every screen and never depends on the pipeline having run.
 *
 * Hospital lookup opens a plain Google Maps search, which needs no API key
 * and no location permission -- Maps resolves "hospital near me" itself.
 */

import { useState } from 'react';

const MAPS_URL = 'https://www.google.com/maps/search/hospital+near+me';

export default function EmergencyButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="Emergency numbers and nearby hospitals"
        aria-label="Emergency help"
        className="fixed bottom-24 right-4 z-40 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-red-600 to-rose-700 text-xl text-white shadow-glow-red transition hover:brightness-110 focus:outline-none focus:ring-4 focus:ring-red-300 sm:bottom-28 sm:right-6"
      >
        <span aria-hidden="true">🚨</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-4 backdrop-blur-sm sm:items-center"
          onClick={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-label="Emergency help"
            className="w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-2xl animate-fade-in-up dark:bg-slate-900"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-gradient-to-br from-red-600 to-rose-700 px-5 py-4 text-white">
              <h2 className="text-lg font-extrabold tracking-wide">Emergency help</h2>
              <p className="mt-0.5 text-sm text-red-50">
                If someone is seriously unwell, call now. Do not wait for this app.
              </p>
            </div>

            <div className="space-y-3 px-5 py-4">
              <a
                href="tel:108"
                className="flex items-center justify-between rounded-xl border border-red-200 bg-red-50 px-4 py-3 transition hover:bg-red-100 dark:border-red-800 dark:bg-red-900/30 dark:hover:bg-red-900/50"
              >
                <span>
                  <span className="block text-lg font-extrabold text-red-700 dark:text-red-300">108</span>
                  <span className="text-xs text-red-700/80 dark:text-red-300/80">Ambulance (India)</span>
                </span>
                <span aria-hidden="true" className="text-xl">📞</span>
              </a>

              <a
                href="tel:112"
                className="flex items-center justify-between rounded-xl border border-red-200 bg-red-50 px-4 py-3 transition hover:bg-red-100 dark:border-red-800 dark:bg-red-900/30 dark:hover:bg-red-900/50"
              >
                <span>
                  <span className="block text-lg font-extrabold text-red-700 dark:text-red-300">112</span>
                  <span className="text-xs text-red-700/80 dark:text-red-300/80">All emergency services (India)</span>
                </span>
                <span aria-hidden="true" className="text-xl">📞</span>
              </a>

              <a
                href={MAPS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between rounded-xl border border-gray-200 px-4 py-3 transition hover:bg-gray-50 dark:border-slate-600 dark:hover:bg-slate-800"
              >
                <span>
                  <span className="block text-sm font-bold text-gray-800 dark:text-gray-100">
                    Find nearby hospitals
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">Opens Google Maps</span>
                </span>
                <span aria-hidden="true" className="text-xl">🏥</span>
              </a>

              <p className="pt-1 text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
                Outside India, dial your local emergency number. If you are unsure
                whether something is an emergency, treat it as one.
              </p>
            </div>

            <div className="border-t border-gray-200 px-5 py-3 text-right dark:border-slate-700">
              <button
                onClick={() => setOpen(false)}
                className="rounded-xl px-4 py-2 text-sm font-medium text-gray-500 transition hover:bg-gray-100 dark:hover:bg-slate-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
