import React from 'react';

interface Props {
  options?: string[];
  onPick: (value: string) => void;
  disabled?: boolean;
  label?: string;
}

const FALLBACK = ['Yes', 'No', 'Sometimes', 'Not sure'];

export default function QuickReplies({ options, onPick, disabled, label = 'Quick reply' }: Props) {
  const replies = options && options.length > 0 ? options : FALLBACK;

  return (
    <div className="flex flex-wrap gap-2 px-1 py-2 animate-fade-in-up">
      <span className="self-center text-xs text-gray-400 dark:text-gray-500 mr-1">{label}:</span>
      {replies.map((r) => (
        <button
          key={r}
          onClick={() => onPick(r)}
          disabled={disabled}
          className="px-3 py-1.5 rounded-full text-xs font-medium border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-gray-300 bg-white dark:bg-slate-800 hover:border-brand-500 hover:text-brand-600 dark:hover:text-brand-300 hover:shadow-sm disabled:opacity-40 transition-all"
        >
          {r}
        </button>
      ))}
    </div>
  );
}
