export default function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 justify-start animate-fade-in-up">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-sm shadow-sm">
        🌿
      </div>
      <div className="bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700 rounded-2xl rounded-bl-md px-4 py-3.5 shadow-soft flex items-center gap-1.5">
        <span className="w-2 h-2 rounded-full bg-brand-400 animate-typing-dot" />
        <span className="w-2 h-2 rounded-full bg-brand-400 animate-typing-dot" style={{ animationDelay: '0.15s' }} />
        <span className="w-2 h-2 rounded-full bg-brand-400 animate-typing-dot" style={{ animationDelay: '0.3s' }} />
        <span className="ml-2 text-xs text-gray-400 dark:text-gray-500">analyzing symptoms…</span>
      </div>
    </div>
  );
}