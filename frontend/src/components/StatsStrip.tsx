import { useEffect, useState } from 'react';
import { api, StatsResponse } from '../api/client';

/**
 * Live facts pulled from the running backend rather than hard-coded, so the
 * numbers on screen are always the model and corpus actually loaded.
 */
export default function StatsStrip() {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    api.stats().then(setStats).catch(() => setStats(null));
  }, []);

  if (!stats) return null;

  const items = [
    { value: stats.diseases, label: 'conditions' },
    { value: stats.symptoms, label: 'symptom features' },
    { value: stats.kb_chunks, label: 'indexed passages' },
    { value: stats.red_flag_patterns, label: 'red-flag rules' },
  ];

  return (
    <div className="mb-8 animate-fade-in-up">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {items.map((it) => (
          <div
            key={it.label}
            className="rounded-xl bg-white/70 dark:bg-slate-800/70 border border-gray-100 dark:border-slate-700 px-3 py-2.5 text-center"
          >
            <div className="text-lg font-extrabold text-gradient tabular-nums">{it.value}</div>
            <div className="text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500">
              {it.label}
            </div>
          </div>
        ))}
      </div>
      <p className="text-center text-[10px] text-gray-400 dark:text-gray-600 mt-2">
        {stats.model_type} · embeddings: {stats.embedding_model} · {stats.llm_mode}
      </p>
    </div>
  );
}
