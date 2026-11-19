interface Props {
  asked: number;
  total: number;
}

/**
 * Shows how far through the intake the user is, so a multi-question
 * consultation reads as deliberate progress rather than an interrogation.
 */
export default function FollowUpProgress({ asked, total }: Props) {
  if (!total || asked <= 0) return null;
  const pct = Math.min(100, Math.round((asked / total) * 100));

  return (
    <div className="flex items-center gap-2 px-1 pt-1 animate-fade-in-up">
      <span className="text-[11px] font-medium text-gray-400 dark:text-gray-500 whitespace-nowrap">
        Consultation · question {asked} of {total}
      </span>
      <div className="flex-1 h-1.5 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-brand-400 to-indigo-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
