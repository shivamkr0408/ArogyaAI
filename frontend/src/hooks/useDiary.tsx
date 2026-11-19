/**
 * Symptom diary: one severity reading per symptom per day.
 *
 * A single consultation is a snapshot. "Is this getting better or worse?"
 * is the question a doctor actually asks, and it needs more than one point.
 * Entries are keyed by (symptom, date) so logging twice in a day corrects
 * the day rather than creating a second point.
 *
 * Device-only, like the rest of the user's data.
 */

import { useCallback, useEffect, useState } from 'react';

const KEY = 'arogya_diary';
const MAX_DAYS = 120;

export interface DiaryEntry {
  symptom: string;
  /** ISO date, no time: one reading per day. */
  date: string;
  /** 0 = gone, 1 = mild, 2 = moderate, 3 = severe. */
  severity: number;
  note?: string;
}

export const SEVERITY_LABELS = ['Gone', 'Mild', 'Moderate', 'Severe'];

export function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function read(): DiaryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function write(entries: DiaryEntry[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    /* quota or private mode */
  }
}

export function useDiary() {
  const [entries, setEntries] = useState<DiaryEntry[]>(read);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setEntries(read());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const log = useCallback((symptom: string, severity: number, note?: string) => {
    setEntries((prev) => {
      const d = today();
      const without = prev.filter((e) => !(e.symptom === symptom && e.date === d));
      const next = [...without, { symptom, date: d, severity, note }]
        .sort((a, b) => (a.date < b.date ? -1 : 1))
        .slice(-MAX_DAYS * 8);
      write(next);
      return next;
    });
  }, []);

  const remove = useCallback((symptom: string) => {
    setEntries((prev) => {
      const next = prev.filter((e) => e.symptom !== symptom);
      write(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => { setEntries([]); write([]); }, []);

  /** Symptoms currently being tracked, most recently updated first. */
  const symptoms = Array.from(new Set(entries.map((e) => e.symptom))).sort((a, b) => {
    const la = entries.filter((e) => e.symptom === a).slice(-1)[0]?.date ?? '';
    const lb = entries.filter((e) => e.symptom === b).slice(-1)[0]?.date ?? '';
    return lb.localeCompare(la);
  });

  const seriesFor = useCallback(
    (symptom: string) =>
      entries.filter((e) => e.symptom === symptom).sort((a, b) => a.date.localeCompare(b.date)),
    [entries],
  );

  /** "improving" | "worsening" | "steady" | null, from first vs last reading. */
  const trendFor = useCallback(
    (symptom: string): string | null => {
      const s = entries
        .filter((e) => e.symptom === symptom)
        .sort((a, b) => a.date.localeCompare(b.date));
      if (s.length < 2) return null;
      const delta = s[s.length - 1].severity - s[0].severity;
      if (delta <= -1) return 'improving';
      if (delta >= 1) return 'worsening';
      return 'steady';
    },
    [entries],
  );

  return { entries, log, remove, clear, symptoms, seriesFor, trendFor };
}
