interface Props {
  onDismiss?: () => void;
  emergencyNote?: string;
}

export default function EmergencyBanner({ onDismiss, emergencyNote }: Props) {
  return (
    <div className="mx-3 mb-2 rounded-2xl bg-gradient-to-br from-red-600 to-rose-700 text-white p-6 shadow-glow-red animate-fade-in-up">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-12 h-12 rounded-full bg-white/15 flex items-center justify-center animate-pulse-ring">
          <span className="text-2xl">🚨</span>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-extrabold mb-1 tracking-wide">MEDICAL EMERGENCY DETECTED</h3>
          <p className="mb-3 leading-relaxed">
            What you're describing could be a medical emergency. Please seek{' '}
            <strong>immediate in-person or emergency medical care now</strong>. Do not rely on
            this tool for emergency situations.
          </p>

          <div className="bg-red-900/40 backdrop-blur rounded-xl px-4 py-3 mb-3 space-y-1">
            {emergencyNote && <p className="text-sm">{emergencyNote}</p>}
            <p className="text-sm">
              <strong>In India:</strong> dial{' '}
              <span className="inline-block bg-white text-red-700 font-mono font-bold px-2 py-0.5 rounded-md mx-0.5">112</span>{' '}
              for emergency services or{' '}
              <span className="inline-block bg-white text-red-700 font-mono font-bold px-2 py-0.5 rounded-md mx-0.5">108</span>{' '}
              for ambulance.
            </p>
            <p className="text-xs text-red-200">
              For other countries, use your local emergency number.
            </p>
          </div>

          {onDismiss && (
            <button
              onClick={onDismiss}
              className="bg-white text-red-600 font-semibold px-5 py-2 rounded-xl hover:bg-red-50 transition shadow-md"
            >
              I understand — return to chat
            </button>
          )}
        </div>
      </div>
    </div>
  );
}