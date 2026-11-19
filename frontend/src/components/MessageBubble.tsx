import { useState } from 'react';
import { useSpeech } from '../hooks/useSpeech';

interface Props {
  content: string;
  role: 'user' | 'assistant';
  animate?: boolean;
}

export default function MessageBubble({ content, role, animate }: Props) {
  const isUser = role === 'user';
  const { supported, speak, speaking, stop } = useSpeech();
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content.replace(/\*\*(.*?)\*\*/g, '$1'));
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked — ignore */
    }
  };

  return (
    <div
      className={`group flex items-end gap-2 ${isUser ? 'justify-end' : 'justify-start'} ${
        animate ? 'animate-fade-in-up' : ''
      }`}
    >
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-sm shadow-sm">
          🌿
        </div>
      )}

      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[82%]`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap shadow-soft ${
            isUser
              ? 'bg-gradient-to-br from-indigo-600 to-violet-600 text-white rounded-br-md'
              : 'bg-white dark:bg-slate-800 text-gray-800 dark:text-gray-100 rounded-bl-md border border-gray-100 dark:border-slate-700'
          }`}
          dangerouslySetInnerHTML={{
            __html: content
              .split('\n')
              .map((line) => line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'))
              .join('<br/>'),
          }}
        />

        {!isUser && (
          <div className="flex items-center gap-1 mt-1 pl-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
            {supported && (
              <button
                onClick={() => (speaking ? stop() : speak(content))}
                title={speaking ? 'Stop reading' : 'Read this aloud'}
                aria-label={speaking ? 'Stop reading' : 'Read this aloud'}
                className="text-[11px] flex items-center gap-1 text-gray-400 hover:text-brand-500 dark:hover:text-brand-300 px-1.5 py-0.5 rounded-md hover:bg-brand-50 dark:hover:bg-slate-700 transition"
              >
                {speaking ? '⏹ Stop' : '🔊 Listen'}
              </button>
            )}
            <button
              onClick={copy}
              title="Copy message"
              aria-label="Copy message"
              className="text-[11px] flex items-center gap-1 text-gray-400 hover:text-brand-500 dark:hover:text-brand-300 px-1.5 py-0.5 rounded-md hover:bg-brand-50 dark:hover:bg-slate-700 transition"
            >
              {copied ? '✓ Copied' : '📋 Copy'}
            </button>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-slate-400 to-slate-600 flex items-center justify-center text-xs text-white shadow-sm">
          You
        </div>
      )}
    </div>
  );
}
