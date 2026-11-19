interface Props {
  doshaImbalance: string;
  diet: string;
  lifestyle: string;
  generalHerbsNote: string;
  caution: string;
}

function Section({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="flex gap-3">
      <span className="flex-shrink-0 text-lg">{icon}</span>
      <div>
        <p className="font-semibold text-sm text-amber-900 dark:text-amber-300">{title}</p>
        <p className="text-sm text-amber-800 dark:text-amber-200/80 leading-relaxed">{text}</p>
      </div>
    </div>
  );
}

export default function AyurvedaCard({ doshaImbalance, diet, lifestyle, generalHerbsNote, caution }: Props) {
  return (
    <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/50 dark:to-orange-950/40 rounded-2xl border border-amber-200/70 dark:border-amber-800/50 p-5 my-2 animate-fade-in-up">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h4 className="font-bold text-amber-900 dark:text-amber-300 flex items-center gap-2 text-sm">
          🌿 Ayurvedic Wellness Guidance
        </h4>
        <span className="text-xs font-semibold px-3 py-1 rounded-full bg-amber-100 dark:bg-amber-900/50 text-amber-800 dark:text-amber-200 border border-amber-200 dark:border-amber-800">
          {doshaImbalance}
        </span>
      </div>

      <div className="space-y-3">
        <Section icon="🥗" title="Dietary guidance" text={diet} />
        <Section icon="🧘" title="Lifestyle suggestions" text={lifestyle} />
        {generalHerbsNote && <Section icon="🌾" title="General herbs note" text={generalHerbsNote} />}
      </div>

      <div className="mt-3 bg-amber-100/70 dark:bg-amber-900/40 rounded-xl px-3 py-2 text-xs text-amber-700 dark:text-amber-300 leading-relaxed">
        ⚠️ {caution}
      </div>
    </div>
  );
}