export default function DisclaimerFooter() {
  return (
    <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur border-t border-gray-200 dark:border-slate-700 px-4 py-2.5 text-center">
      <p className="text-xs text-gray-400 dark:text-gray-500 max-w-3xl mx-auto">
        <strong className="text-gray-500 dark:text-gray-400">Disclaimer:</strong> Arogya AI is an
        educational, informational tool and is <strong>not</strong> a medical diagnosis. Always
        consult a licensed healthcare professional for medical advice, diagnosis, or treatment.
        In a medical emergency, call your local emergency number.
      </p>
    </div>
  );
}