import React from 'react';
import StatsStrip from './StatsStrip';

const FEATURES = [
  {
    icon: '🩺',
    title: 'Symptom Analysis',
    desc: 'Tell us how you feel in plain words — we extract symptoms intelligently.',
  },
  {
    icon: '🚨',
    title: 'Red-Flag Safety First',
    desc: 'Every message is scanned for emergency warning signs before anything else.',
  },
  {
    icon: '🧠',
    title: 'AI Prediction',
    desc: 'A Bernoulli Naive Bayes model ranks the most likely conditions.',
  },
  {
    icon: '❓',
    title: 'Guided Follow-Ups',
    desc: 'A structured intake asks about duration, intensity and trend before concluding.',
  },
  {
    icon: '🧾',
    title: 'Shareable Summary',
    desc: 'Export the whole consultation as a report you can take to a doctor.',
  },
  {
    icon: '🪴',
    title: 'Ayurvedic Guidance',
    desc: 'Gentle, non-prescriptive diet & lifestyle advice from a curated knowledge base.',
  },
  {
    icon: '📚',
    title: 'Evidence Grounded',
    desc: 'Explanations are backed by a local RAG knowledge base — no invented claims.',
  },
  {
    icon: '🎙️',
    title: 'Talk & Listen',
    desc: 'Speak your symptoms and have the guidance read back aloud — full two-way voice.',
  },
  {
    icon: '🔒',
    title: 'Private & Anonymous',
    desc: 'No login, no cloud storage — your session stays on your machine.',
  },
];

const DIFFERENTIATORS = [
  'Safety-first',
  'RAG-grounded',
  'Explainable',
  'Ayurvedic',
  'Voice I/O',
  'Private',
];

const SUGGESTIONS = [
  "I've had a fever and headache for two days",
  'My knees hurt and they feel swollen and stiff',
  "I can't stop coughing and my chest feels tight",
];

interface Props {
  onSuggestion: (text: string) => void;
  /** Open "I know my condition" mode, optionally seeded with typed text. */
  onKnowCondition: (seed?: string) => void;
}

export default function WelcomeScreen({ onSuggestion, onKnowCondition }: Props) {
  return (
    <div className="flex-1 overflow-y-auto overflow-x-hidden">
      <div className="relative overflow-hidden max-w-3xl mx-auto px-4 py-10 animate-fade-in-up">
        <div className="aurora" aria-hidden="true" />
        <div className="relative text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-500 to-indigo-600 shadow-glow mb-4">
            <span className="text-3xl">🌿</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-800 dark:text-white">
            Arogya <span className="text-gradient">AI</span>
          </h1>
          <p className="mt-2 text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            Your conversational health-guidance companion. Describe your symptoms and get
            informed, educational guidance — never a diagnosis.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-1.5">
            {DIFFERENTIATORS.map((d) => (
              <span
                key={d}
                className="text-[10px] font-semibold tracking-wide px-2.5 py-1 rounded-full bg-white/70 dark:bg-slate-800/70 border border-brand-100 dark:border-slate-700 text-brand-700 dark:text-brand-300"
              >
                {d}
              </span>
            ))}
          </div>
        </div>


        {/* Two ways in. Most people arrive already knowing their diagnosis,
            and asking them to describe symptoms they have had explained to
            them is the wrong first question. */}
        <div className="mb-6 grid gap-3 sm:grid-cols-2">
          <button
            onClick={() => onSuggestion("I've had a fever and headache for two days")}
            className="group rounded-2xl border border-brand-200 bg-white/70 p-4 text-left transition hover:border-brand-400 hover:shadow-soft dark:border-slate-700 dark:bg-slate-800/70"
          >
            <div className="text-2xl">🩺</div>
            <h3 className="mt-1.5 font-bold text-gray-800 dark:text-white">Check my symptoms</h3>
            <p className="mt-0.5 text-xs leading-relaxed text-gray-500 dark:text-gray-400">
              Describe how you feel. I'll ask a few questions and show a ranked
              differential with the reasoning.
            </p>
          </button>

          <button
            onClick={() => onKnowCondition()}
            className="group rounded-2xl border border-indigo-200 bg-white/70 p-4 text-left transition hover:border-indigo-400 hover:shadow-soft dark:border-slate-700 dark:bg-slate-800/70"
          >
            <div className="text-2xl">📖</div>
            <h3 className="mt-1.5 font-bold text-gray-800 dark:text-white">I know my condition</h3>
            <p className="mt-0.5 text-xs leading-relaxed text-gray-500 dark:text-gray-400">
              Already diagnosed? Ask about diet, contagiousness, recovery time
              and when to worry.
            </p>
          </button>
        </div>

        <StatsStrip />

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-8">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-gray-100 dark:border-slate-700 shadow-soft hover:-translate-y-0.5 transition-transform"
            >
              <div className="text-2xl mb-1">{f.icon}</div>
              <h3 className="font-semibold text-sm text-gray-800 dark:text-white">{f.title}</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>

        <div>
          <p className="text-center text-xs uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">
            Try one of these
          </p>
          <div className="flex flex-col gap-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => onSuggestion(s)}
                className="text-left bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl px-4 py-3 text-sm text-gray-700 dark:text-gray-200 hover:border-brand-400 hover:shadow-glow transition-all"
              >
                “{s}”
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}