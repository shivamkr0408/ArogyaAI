import { conditionApi } from '../api/client';
import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChatSession';
import { useTheme } from '../hooks/useTheme';
import { useSpeech } from '../hooks/useSpeech';
import MessageBubble from './MessageBubble';
import SymptomChips from './SymptomChips';
import PredictionCard from './PredictionCard';
import AyurvedaCard from './AyurvedaCard';
import EmergencyBanner from './EmergencyBanner';
import DisclaimerFooter from './DisclaimerFooter';
import AboutPanel from './AboutPanel';
import WelcomeScreen from './WelcomeScreen';
import QuickReplies from './QuickReplies';
import TypingIndicator from './TypingIndicator';
import FollowUpProgress from './FollowUpProgress';
import PipelinePanel from './PipelinePanel';
import TriageCard from './TriageCard';
import DifferentialChart from './DifferentialChart';
import ReportModal from './ReportModal';
import NlpLabPanel from './NlpLabPanel';
import EvaluationPanel from './EvaluationPanel';
import ExplanationCard from './ExplanationCard';
import NlpTurnPanel from './NlpTurnPanel';
import HistoryPanel from './HistoryPanel';
import BodyMap from './BodyMap';
import WhyThisQuestion from './WhyThisQuestion';
import ProfilePanel from './ProfilePanel';
import EmergencyButton from './EmergencyButton';
import SymptomDiary from './SymptomDiary';
import DoctorReport from './DoctorReport';
import ConditionMode from './ConditionMode';

type Meta = Record<string, any>;

function metaOf(m: { role: string; content: string; metadata_?: unknown }): Meta | undefined {
  return (m.metadata_ as Meta) || undefined;
}

export default function ChatWindow() {
  const { messages, state, loading, send, startNew, sessionId } = useChat();
  // "I know my condition" mode runs alongside the symptom chat rather than
  // inside it, so the existing state machine is untouched.
  const [profileOpen, setProfileOpen] = useState(false);
  const [diaryOpen, setDiaryOpen] = useState(false);
  const [doctorOpen, setDoctorOpen] = useState(false);
  // Symptoms seen in this consultation, offered as diary shortcuts.
  const lastSymptoms: string[] = Array.from(
    new Set(
      messages.flatMap((mm) => (metaOf(mm)?.extracted_symptoms as string[] | undefined) ?? []),
    ),
  );
  const [conditionMode, setConditionMode] = useState(false);
  const [conditionSeed, setConditionSeed] = useState<string | undefined>(undefined);
  const { dark, toggle } = useTheme();
  const { supported: ttsSupported, narrate, setNarrate, speak } = useSpeech();
  const spokenRef = useRef<string>('');
  const [input, setInput] = useState('');
  const [emergencyDismissed, setEmergencyDismissed] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [nlpLabOpen, setNlpLabOpen] = useState(false);
  const [evalOpen, setEvalOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isEmergency = state === 'emergency' && !emergencyDismissed;
  const isClarifying = state === 'clarifying' && !loading;

  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const lastMeta = lastAssistant ? metaOf(lastAssistant) : undefined;
  const suggested: string[] = lastMeta?.suggested_replies || [];
  const showSuggestions = !loading && !isEmergency && suggested.length > 0;
  const hasResult = messages.some((m) => metaOf(m)?.state === 'result');

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto read-aloud the newest assistant reply when narration is on.
  useEffect(() => {
    if (!narrate || loading || messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.role !== 'assistant') return;
    const key = `${messages.length}:${last.content.slice(0, 48)}`;
    if (spokenRef.current === key) return;
    spokenRef.current = key;
    speak(last.content);
  }, [messages, loading, narrate, speak]);

  const startVoice = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.lang = 'en-IN';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => setListening(true);
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    rec.onresult = (e: any) => {
      const transcript = e.results[0][0].transcript;
      setInput(transcript);
    };
    try {
      rec.start();
    } catch {
      setListening(false);
    }
  };

  // "I have diabetes" is a statement of diagnosis, not a complaint. Offer
  // condition mode instead of running it through the symptom checker, but
  // only when the sentence both reads as a declaration AND names something
  // actually recognised -- "I have a headache" must stay in symptom mode.
  const DECLARATION_RE =
    /\b(?:i\s+have|i've\s+got|diagnosed\s+with|doctor\s+(?:said|told)|suffering\s+from|known\s+case\s+of)\b|\bmujhe\b.*\bhai\b/i;

  const maybeConditionDeclaration = async (value: string): Promise<boolean> => {
    if (!DECLARATION_RE.test(value)) return false;
    try {
      const r = await conditionApi.match(value);
      if (r.status === 'matched' && r.condition) {
        setConditionSeed(value);
        setConditionMode(true);
        return true;
      }
    } catch {
      /* offline or server down -- fall through to the normal symptom path */
    }
    return false;
  };

  const handleSend = async (text?: string) => {
    const value = (text ?? input).trim();
    if (!value || loading || isEmergency) return;
    setInput('');
    if (!hasMessages && (await maybeConditionDeclaration(value))) return;
    send(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const reset = () => {
    startNew();
    setEmergencyDismissed(false);
    setDetailsOpen(false);
    setReportOpen(false);
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col h-screen overflow-x-hidden bg-gradient-to-b from-gray-50 to-brand-50/40 dark:from-slate-900 dark:to-slate-900">
      <header className="glass bg-white/70 dark:bg-slate-900/70 border-b border-gray-200/70 dark:border-slate-700 px-4 py-3 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-xl shadow-soft">
            🌿
          </div>
          <div>
            <h1 className="text-lg font-extrabold text-gray-800 dark:text-white leading-tight">
              Arogya <span className="text-gradient">AI</span>
            </h1>
            <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-tight">
              Conversational health guidance
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <span
            className="hidden sm:inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 dark:text-gray-400 mr-1"
            title="Backend + ML model + vector store online"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Online
          </span>

          {ttsSupported && (
            <button
              onClick={() => setNarrate(!narrate)}
              aria-label="Toggle read-aloud"
              title={narrate ? 'Read replies aloud: ON' : 'Read replies aloud: OFF'}
              className={`w-9 h-9 rounded-lg flex items-center justify-center text-lg transition ${
                narrate
                  ? 'bg-brand-100 dark:bg-brand-900/40 text-brand-600 dark:text-brand-300'
                  : 'hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-400'
              }`}
            >
              {narrate ? '🔊' : '🔇'}
            </button>
          )}

          <button
            onClick={toggle}
            aria-label="Toggle dark mode"
            className="w-9 h-9 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-700 flex items-center justify-center text-lg transition"
          >
            {dark ? '☀️' : '🌙'}
          </button>

          {hasResult && sessionId && (
            <button
              onClick={() => setReportOpen(true)}
              title="Export this consultation as a summary"
              className="text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-700 px-3 py-1.5 rounded-lg transition"
            >
              🧾 <span className="hidden sm:inline">Summary</span>
            </button>
          )}

          {hasResult && sessionId && (
            <button
              onClick={() => setDoctorOpen(true)}
              title="A printable report to take to a doctor"
              className="text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-700 px-2.5 sm:px-3 py-1.5 rounded-lg transition"
            >
              🩺 <span className="hidden md:inline">Doctor report</span>
            </button>
          )}

          <button
            onClick={() => setNlpLabOpen(true)}
            title="Run any sentence through the NLP pipeline and see every stage"
            className="text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-700 px-2.5 sm:px-3 py-1.5 rounded-lg transition"
          >
            🔬 <span className="hidden md:inline">NLP Lab</span>
          </button>

          <button
            onClick={() => setEvalOpen(true)}
            title="Measured accuracy of every component"
            className="text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-700 px-2.5 sm:px-3 py-1.5 rounded-lg transition"
          >
            📊 <span className="hidden md:inline">Evaluation</span>
          </button>

          <button
            onClick={() => { setConditionSeed(undefined); setConditionMode(true); }}
            title="Ask about a condition you already have"
            className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-gray-600 transition hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-slate-700 sm:px-3"
          >
            📖 <span className="hidden md:inline">My condition</span>
          </button>

          <button
            onClick={() => setProfileOpen(true)}
            title="Health profile - makes the safety checks person-aware"
            className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-gray-600 transition hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-slate-700 sm:px-3"
          >
            👤 <span className="hidden md:inline">Profile</span>
          </button>

          <button
            onClick={() => setDiaryOpen(true)}
            title="Track a symptom day by day"
            className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-gray-600 transition hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-slate-700 sm:px-3"
          >
            📈 <span className="hidden md:inline">Diary</span>
          </button>

          <HistoryPanel />

          <AboutPanel />

          <button
            onClick={reset}
            title="Start a new consultation"
            className="text-sm font-medium text-brand-600 dark:text-brand-300 hover:bg-brand-50 dark:hover:bg-brand-900/30 px-2.5 sm:px-3 py-1.5 rounded-lg transition"
          >
            ↻ <span className="hidden sm:inline">New chat</span>
          </button>
        </div>
      </header>

      {conditionMode ? (
        <ConditionMode
          initialText={conditionSeed}
          onClose={() => setConditionMode(false)}
          onSwitchToSymptoms={(seed) => {
            // Bridge back into the symptom checker, carrying the context.
            setConditionMode(false);
            if (seed) handleSend(seed);
          }}
        />
      ) : hasMessages ? (
        <div className="flex-1 overflow-y-auto px-4 py-5">
          <div className="max-w-3xl mx-auto space-y-3">
            {messages.map((msg, i) => {
              const m = metaOf(msg);
              const last = i === messages.length - 1;
              return (
                <React.Fragment key={i}>
                  <MessageBubble content={msg.content} role={msg.role} animate={last} />
                  {msg.role === 'assistant' && m && (
                    <>
                      <SymptomChips symptoms={m.extracted_symptoms || []} />
                      {m.state === 'clarifying' && last && (
                        <>
                          <FollowUpProgress
                            asked={m.followup_asked || 0}
                            total={m.followup_total || 0}
                          />
                          <WhyThisQuestion
                            informationGain={m.information_gain}
                            rationale={m.question_rationale}
                          />
                        </>
                      )}
                      {m.state === 'result' && m.prediction && (
                        <>
                          {m.prediction.abstained && (
                            <div className="my-2 rounded-2xl border border-amber-200 bg-amber-50/70 p-3 dark:border-amber-800 dark:bg-amber-900/20">
                              <p className="text-sm font-semibold text-amber-900 dark:text-amber-200">
                                Not enough to name one condition
                              </p>
                              <p className="mt-0.5 text-xs leading-relaxed text-amber-800 dark:text-amber-300/90">
                                {m.prediction.abstain_reason === 'too_few_symptoms'
                                  ? 'I only have one symptom to go on, which is not enough to point at a single condition. The possibilities below are ranked, but treat them as a starting point and tell me more.'
                                  : 'The leading possibility is not clearly ahead of the others at this level of evidence, so I would rather show you the range than name one. Telling me more would sharpen it.'}
                              </p>
                            </div>
                          )}
                          <PredictionCard
                            confidence={m.prediction.confidence}
                            condition={m.prediction.top_condition}
                            severity={m.prediction.severity}
                            alternatives={m.prediction.alternatives}
                            reasoning={m.prediction.reasoning_summary}
                            evidenceSources={m.prediction.evidence_sources}
                            showDetails={detailsOpen}
                            onToggle={() => setDetailsOpen(!detailsOpen)}
                            disclaimer={m.prediction.disclaimer}
                            differential={
                              <DifferentialChart
                                top={{
                                  disease: m.prediction.top_condition,
                                  probability: m.prediction.confidence,
                                }}
                                alternatives={m.prediction.alternatives || []}
                              />
                            }
                          />
                          {m.prediction.triage && (
                            <TriageCard
                              triage={m.prediction.triage}
                              watchFor={m.prediction.watch_for}
                              clinicalContext={m.prediction.clinical_context}
                            />
                          )}
                          {m.prediction.ayurvedic_guidance && (
                            <AyurvedaCard {...m.prediction.ayurvedic_guidance} />
                          )}
                          {m.prediction.explanation && (
                            <ExplanationCard explanation={m.prediction.explanation} />
                          )}
                        </>
                      )}
                      {m.nlp && (
                        <NlpTurnPanel
                          nlp={m.nlp}
                          userText={messages[i - 1]?.role === 'user' ? messages[i - 1].content : ''}
                          deniedSymptoms={m.denied_symptoms || []}
                        />
                      )}
                      {last && <PipelinePanel stages={m.pipeline || []} />}
                    </>
                  )}
                </React.Fragment>
              );
            })}

            {loading && <TypingIndicator />}
            <div ref={bottomRef} />
          </div>
        </div>
      ) : (
        <WelcomeScreen
          onSuggestion={(t) => handleSend(t)}
          onKnowCondition={(seed) => { setConditionSeed(seed); setConditionMode(true); }}
        />
      )}

      {isEmergency && (
        <EmergencyBanner
          emergencyNote={metaOf(messages[messages.length - 1])?.emergency_numbers_note}
          onDismiss={() => setEmergencyDismissed(true)}
        />
      )}

      {/* Condition mode has its own ask box; showing the symptom composer
          as well would put two different inputs on screen at once. */}
      {!conditionMode && (
      <div className="border-t border-gray-200/70 dark:border-slate-700 bg-white/80 dark:bg-slate-900/70 backdrop-blur px-4 pt-2 pb-3">
        <div className="max-w-3xl mx-auto">
          {showSuggestions && (
            <QuickReplies
              options={suggested}
              onPick={(v) => handleSend(v)}
              disabled={loading}
              label={isClarifying ? 'Quick reply' : 'Try asking'}
            />
          )}

          <div className="flex items-center gap-2 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-600 rounded-2xl px-2 py-2 shadow-soft focus-within:border-brand-400 focus-within:ring-2 focus-within:ring-brand-200 dark:focus-within:ring-brand-900/50 transition">
            <button
              onClick={startVoice}
              disabled={isEmergency}
              title="Voice input (experimental)"
              aria-label="Voice input"
              className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition ${
                listening
                  ? 'mic-recording bg-red-100 dark:bg-red-900/50 text-red-500'
                  : 'text-gray-400 hover:text-brand-500 hover:bg-brand-50 dark:hover:bg-slate-700'
              }`}
            >
              {listening ? '⏺' : '🎤'}
            </button>

            <BodyMap
              onCompose={(text) =>
                // Append rather than replace, so a picked symptom can be
                // added to something already half-typed.
                setInput((prev) => (prev.trim() ? `${prev.trim()}, ${text.replace(/^I have /, '')}` : text))
              }
            />

            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isEmergency
                  ? 'Emergency mode — please seek immediate care'
                  : listening
                    ? 'Listening… speak now'
                    : 'Describe your symptoms… e.g. "fever and headache since yesterday"'
              }
              disabled={isEmergency || loading || listening}
              className="flex-1 bg-transparent px-2 text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none disabled:opacity-50"
            />

            <button
              onClick={() => handleSend()}
              disabled={isEmergency || loading || !input.trim()}
              className="flex-shrink-0 bg-gradient-to-r from-brand-500 to-indigo-600 text-white px-4 py-2 rounded-xl text-sm font-semibold hover:shadow-glow hover:brightness-105 active:scale-95 transition disabled:from-gray-200 disabled:to-gray-300 disabled:text-gray-400 dark:disabled:from-slate-700 dark:disabled:to-slate-700 disabled:cursor-not-allowed disabled:shadow-none"
            >
              Send ➤
            </button>
          </div>

          <p className="text-center text-[10px] text-gray-400 dark:text-gray-600 mt-2">
            Educational tool only · not a medical diagnosis
          </p>
        </div>
      </div>
      )}

      <DisclaimerFooter />

      {reportOpen && sessionId && (
        <ReportModal sessionId={sessionId} onClose={() => setReportOpen(false)} />
      )}

      {nlpLabOpen && <NlpLabPanel onClose={() => setNlpLabOpen(false)} />}

      {evalOpen && <EvaluationPanel onClose={() => setEvalOpen(false)} />}

      {profileOpen && <ProfilePanel onClose={() => setProfileOpen(false)} />}

      {doctorOpen && sessionId && (
        <DoctorReport sessionId={sessionId} onClose={() => setDoctorOpen(false)} />
      )}

      {diaryOpen && (
        <SymptomDiary
          onClose={() => setDiaryOpen(false)}
          suggested={lastSymptoms}
        />
      )}

      <EmergencyButton />
    </div>
  );
}