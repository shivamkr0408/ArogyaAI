import React, { useState } from 'react';

export default function AboutPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700 px-3 py-1.5 rounded-lg transition"
        aria-label="About this project"
      >
        About
      </button>

      {open && (
        <div className="absolute right-0 top-10 w-80 max-w-[calc(100vw-1.5rem)] bg-white dark:bg-slate-800 rounded-xl shadow-soft border border-gray-100 dark:border-slate-700 p-5 z-50">
          <h3 className="font-bold text-gray-800 dark:text-white mb-2">About Arogya AI</h3>
          <div className="text-xs text-gray-600 dark:text-gray-300 space-y-2">
            <p>
              A conversational health-guidance assistant that uses an NLP pipeline to understand
              symptoms and provide educational, non-diagnostic guidance.
            </p>
            <p className="font-medium text-gray-700 dark:text-gray-200">Pipeline:</p>
            <ul className="list-disc list-inside">
              <li>LLM / fuzzy symptom extraction from free text</li>
              <li>Emergency red-flag safety layer — every turn</li>
              <li>Bernoulli Naive Bayes classifier (102 symptoms, 41 conditions)</li>
              <li>RAG grounding from a local vector knowledge base</li>
              <li>Curated Ayurvedic lifestyle & diet guidance</li>
            </ul>
            <div className="bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg px-3 py-2 text-amber-700 dark:text-amber-300">
              Arogya AI never diagnoses or prescribes. Educational tool only — consult a licensed
              healthcare professional.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}