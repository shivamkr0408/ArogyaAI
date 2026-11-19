import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from 'react';

/**
 * Text-to-speech ("read aloud") for assistant replies — the output half of the
 * voice loop (voice input already lives in ChatWindow). Browser-native via the
 * Web Speech API, so no dependency and no backend call. Accessibility win for
 * low-literacy / hands-free use, in line with the doctor-like consultation goal.
 */
interface SpeechCtx {
  supported: boolean;
  narrate: boolean;
  setNarrate: (v: boolean) => void;
  speaking: boolean;
  speak: (text: string) => void;
  stop: () => void;
}

const Ctx = createContext<SpeechCtx | undefined>(undefined);

/** Strip markdown / emoji noise so the voice reads clean prose. */
function cleanForSpeech(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/[#*_`>~|]/g, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function pickVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  return (
    voices.find((v) => /en-IN/i.test(v.lang)) ||
    voices.find((v) => /en-GB/i.test(v.lang)) ||
    voices.find((v) => /^en/i.test(v.lang)) ||
    voices[0]
  );
}

export function SpeechProvider({ children }: { children: ReactNode }) {
  const supported =
    typeof window !== 'undefined' && 'speechSynthesis' in window;
  const [narrate, setNarrateState] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    try {
      setNarrateState(localStorage.getItem('arogya-narrate') === '1');
    } catch {
      /* ignore */
    }
    // Warm up the voice list (some browsers load it async).
    if (supported) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () =>
        window.speechSynthesis.getVoices();
    }
  }, [supported]);

  const stop = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  const setNarrate = useCallback(
    (v: boolean) => {
      setNarrateState(v);
      try {
        localStorage.setItem('arogya-narrate', v ? '1' : '0');
      } catch {
        /* ignore */
      }
      if (!v) stop();
    },
    [stop]
  );

  const speak = useCallback(
    (text: string) => {
      if (!supported) return;
      const clean = cleanForSpeech(text);
      if (!clean) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(clean);
      u.lang = 'en-IN';
      u.rate = 1.02;
      u.pitch = 1;
      const voice = pickVoice();
      if (voice) u.voice = voice;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(u);
    },
    [supported]
  );

  useEffect(
    () => () => {
      if (supported) window.speechSynthesis.cancel();
    },
    [supported]
  );

  return (
    <Ctx.Provider
      value={{ supported, narrate, setNarrate, speaking, speak, stop }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useSpeech(): SpeechCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error('useSpeech must be used within SpeechProvider');
  return c;
}
