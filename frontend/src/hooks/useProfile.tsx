/**
 * Health profile, held on this device only.
 *
 * It is sent with each turn so the backend can apply person-dependent
 * red-flag rules -- a fever is routine in a healthy adult and an emergency
 * in a baby under one -- and is never persisted server-side. That is the
 * reason it lives in localStorage rather than in the session row: the
 * privacy claim on the welcome screen has to stay true.
 */

import { useCallback, useEffect, useState } from 'react';

const KEY = 'arogya_profile';

export type AgeGroup = 'under_1' | '1_12' | '13_17' | '18_64' | '65_plus';

export interface HealthProfile {
  age_group?: AgeGroup;
  sex?: 'female' | 'male' | 'other' | 'prefer_not_to_say';
  pregnant: boolean;
  known_conditions: string[];
  allergies: string[];
}

export const EMPTY_PROFILE: HealthProfile = {
  pregnant: false,
  known_conditions: [],
  allergies: [],
};

export const AGE_LABELS: Record<AgeGroup, string> = {
  under_1: 'Under 1 year',
  '1_12': '1 to 12',
  '13_17': '13 to 17',
  '18_64': '18 to 64',
  '65_plus': '65 or over',
};

function read(): HealthProfile {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return EMPTY_PROFILE;
    return { ...EMPTY_PROFILE, ...JSON.parse(raw) };
  } catch {
    return EMPTY_PROFILE;
  }
}

export function useProfile() {
  const [profile, setProfile] = useState<HealthProfile>(read);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setProfile(read());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const save = useCallback((next: HealthProfile) => {
    setProfile(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* private mode: the profile simply does not persist */
    }
  }, []);

  const update = useCallback(
    (patch: Partial<HealthProfile>) => save({ ...read(), ...patch }),
    [save],
  );

  /** Remove everything this app has stored on the device. */
  const clearAll = useCallback(() => {
    try {
      localStorage.removeItem(KEY);
      localStorage.removeItem('arogya_consultations');
      localStorage.removeItem('arogya_diary');
      localStorage.removeItem('arogya_session_id');
    } catch {
      /* nothing to clear */
    }
    setProfile(EMPTY_PROFILE);
  }, []);

  /** Only send fields the user actually filled in. */
  const forRequest = useCallback((): HealthProfile | undefined => {
    const p = profile;
    const meaningful =
      p.age_group || p.sex || p.pregnant ||
      p.known_conditions.length || p.allergies.length;
    return meaningful ? p : undefined;
  }, [profile]);

  const isSet = Boolean(
    profile.age_group || profile.sex || profile.pregnant ||
    profile.known_conditions.length || profile.allergies.length,
  );

  return { profile, update, save, clearAll, forRequest, isSet };
}
