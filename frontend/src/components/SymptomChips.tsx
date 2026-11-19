interface Props {
  symptoms: string[];
}

export default function SymptomChips({ symptoms }: Props) {
  if (!symptoms.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 my-1.5 animate-fade-in-up">
      {symptoms.map((s) => (
        <span
          key={s}
          className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-brand-50 dark:bg-brand-900/40 text-brand-700 dark:text-brand-300 border border-brand-100 dark:border-brand-800"
        >
          <span className="w-1 h-1 rounded-full bg-brand-400" />
          {s.replace(/_/g, ' ')}
        </span>
      ))}
    </div>
  );
}