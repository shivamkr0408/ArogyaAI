/**
 * Local record of past consultations.
 *
 * Until now "New chat" called createSession() and overwrote the single
 * session id in localStorage, so the previous consultation became
 * unreachable even though the backend still had every turn of it. That is
 * the wrong default for a health tool: the useful question is usually "what
 * did it say last week, and am I better or worse now?".
 *
 * Only an index is kept on the device -- id, when, leading condition,
 * confidence, triage level. The transcript itself is never copied here; it
 * stays server-side and is re-fetched through the existing
 * /chat/history/{id} endpoint when a consultation is reopened. So this adds
 * a way back to a consultation without duplicating any clinical content.
 */

import { useCallback, useEffect, useState } from 'react';
import type { ChatResponse } from '../api/client';

const KEY = 'arogya_consultations';
const LIMIT = 30;

export interface ConsultationRecord {
  id: string;
  startedAt: string;
  updatedAt: string;
  /** First thing the user said, used as the list title. */
  opening: string;
  turns: number;
  condition?: string;
  confidence?: number;
  severity?: string;
  triageLevel?: string;
  triageLabel?: string;
  /** True when the consultation ended in the terminal emergency state. */
  emergency?: boolean;
}

function read(): ConsultationRecord[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ConsultationRecord[]) : [];
  } catch {
    // Corrupt or blocked storage is not worth breaking the app over.
    return [];
  }
}

function write(records: ConsultationRecord[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(records.slice(0, LIMIT)));
  } catch {
    /* private mode / quota -- history is a convenience, not a requirement */
  }
}

export function useHistory() {
  const [records, setRecords] = useState<ConsultationRecord[]>(read);

  // Keep multiple open tabs roughly in step.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setRecords(read());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const upsert = useCallback((sessionId: string, patch: Partial<ConsultationRecord>) => {
    setRecords(prev => {
      const now = new Date().toISOString();
      const idx = prev.findIndex(r => r.id === sessionId);
      let next: ConsultationRecord[];
      if (idx === -1) {
        next = [{
          id: sessionId,
          startedAt: now,
          updatedAt: now,
          opening: patch.opening ?? 'Consultation',
          turns: patch.turns ?? 0,
          ...patch,
        }, ...prev];
      } else {
        next = [...prev];
        next[idx] = { ...next[idx], ...patch, updatedAt: now };
        // Most recently touched first.
        const [moved] = next.splice(idx, 1);
        next.unshift(moved);
      }
      write(next);
      return next;
    });
  }, []);

  /** Fold a chat response into the record for this session. */
  const recordTurn = useCallback((
    sessionId: string,
    res: ChatResponse,
    firstUserMessage?: string,
  ) => {
    const patch: Partial<ConsultationRecord> = { turns: res.turn_count };
    if (firstUserMessage) patch.opening = firstUserMessage;
    if (res.state === 'emergency') patch.emergency = true;
    if (res.prediction) {
      patch.condition = res.prediction.top_condition;
      patch.confidence = res.prediction.confidence;
      patch.severity = res.prediction.severity;
      if (res.prediction.triage) {
        patch.triageLevel = res.prediction.triage.level;
        patch.triageLabel = res.prediction.triage.label;
      }
    }
    upsert(sessionId, patch);
  }, [upsert]);

  const remove = useCallback((id: string) => {
    setRecords(prev => {
      const next = prev.filter(r => r.id !== id);
      write(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setRecords([]);
    write([]);
  }, []);

  return { records, recordTurn, upsert, remove, clear };
}
