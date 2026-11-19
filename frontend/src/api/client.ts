const API_BASE = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export interface PipelineStage {
  name: string;
  status: 'ok' | 'skipped' | 'triggered';
  detail: string;
  ms: number;
}

export interface Assertion {
  label: 'affirmed' | 'negated' | 'family' | 'hypothetical' | 'historical' | 'uncertain';
  negation: string;
  temporality: string;
  experiencer: string;
  certainty: string;
  is_clinically_present: boolean;
  cues: { text: string; kind: string; direction: string; char_start: number; char_end: number }[];
}

export interface Entity {
  label: 'SYMPTOM' | 'DISEASE' | 'SEVERITY' | 'BODY_SITE' | 'TEMPORAL';
  text: string;
  original_text: string;
  canonical: string;
  char_start: number;
  char_end: number;
  strategy: string;
  confidence: number;
  severity?: string | null;
  assertion?: Assertion | null;
  is_present: boolean;
}

export interface NormalizationEdit {
  kind: 'elongation' | 'contraction' | 'code_mixed' | 'spelling';
  original: string;
  replacement: string;
  orig_start: number;
  orig_end: number;
}

export interface Distress {
  score: number;
  level: 'calm' | 'elevated' | 'high';
  triggers: string[];
  needs_acknowledgement: boolean;
}

export interface IntentResult {
  intent: string;
  confidence: number;
  source: string;
  scores: Record<string, number>;
}

export interface NlpTrace {
  normalized: string;
  entities: Entity[];
  normalization_edits: NormalizationEdit[];
  intent: IntentResult;
  distress: Distress;
  duration_days?: number | null;
  duration_bucket?: string | null;
  trend?: string | null;
  frequency?: string | null;
  severity?: string | null;
  timings: { name: string; ms: number; detail: string }[];
  total_ms: number;
}

export interface SymptomContribution {
  symptom: string;
  label: string;
  log_likelihood: number;
  discrimination?: number | null;
  support_share: number;
}

export interface Explanation {
  condition: string;
  compared_with?: string | null;
  model: string;
  weight_meaning: string;
  log_prior?: number | null;
  contributions: SymptomContribution[];
  top_discriminating: SymptomContribution[];
  denial_evidence: { label: string; log_absence_likelihood: number; discrimination?: number | null }[];
  expected_but_not_reported: { label: string; log_likelihood: number }[];
  note: string;
}

export interface EvidenceChunk {
  source_file: string;
  disease_slug: string;
  similarity?: number | null;
  excerpt: string;
}

export interface NlpAnalysis extends NlpTrace {
  original: string;
  tokens: { text: string; stem: string; start: number; end: number; sentence: number }[];
  sentences: string[];
  temporal: { kind: string; text: string; char_start: number; char_end: number; days?: number | null; normalized?: string | null }[];
  present_symptoms: string[];
  negated_symptoms: string[];
  conditions: { condition: string; attribution: string; status: string; text: string }[];
  cues: {
    text: string; kind: string; direction: string; sentence: number;
    char_start: number; char_end: number; scope_text: string;
    scope_start: number | null; scope_end: number | null;
  }[];
}

export interface ModelResult {
  model: string;
  note: string;
  accuracy: number;
  top3_accuracy: number | null;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_f1: number;
  cv_accuracy_mean: number | null;
  cv_accuracy_std: number | null;
  fit_ms: number;
  predict_ms_per_1000: number;
  supports_probability: boolean;
}

export interface EvaluationResponse {
  deployed_model: { estimator: string | null; metrics: Record<string, any> };
  variants: Record<string, any>;
  model_comparison: {
    generated_at: string;
    dataset: { train_rows: number; test_rows: number; features: number; classes: number; cv_folds: number };
    results: ModelResult[];
    best_by_macro_f1: string;
    deployed_model: string;
    macro_f1_gap_to_best: number;
    selection_rationale: string;
    notes?: string[];
  } | null;
  nlp_components: Record<string, any> | null;
  retrieval: Record<string, any> | null;
  missing: string[];
  how_to_regenerate: Record<string, string>;
}

export interface NlpCapabilities {
  stages: { name: string; algorithms: string[]; detail: string }[];
  resources: Record<string, number>;
}

export interface ChatResponse {
  reply_text: string;
  state: string;
  extracted_symptoms: string[];
  denied_symptoms: string[];
  intent?: string | null;
  nlp?: NlpTrace | null;
  turn_count: number;
  prediction?: Prediction;
  matched_pattern?: string;
  emergency_numbers_note?: string;
  suggested_replies: string[];
  question_topic?: string;
  /** Why this question was chosen (expected information gain). */
  question_rationale?: string | null;
  information_gain?: number | null;
  profile_concerns?: { level: string; rule: string; title: string; detail: string }[];
  followup_asked: number;
  followup_total: number;
  pipeline: PipelineStage[];
}

export interface Triage {
  level: 'self_care' | 'soon' | 'urgent';
  label: string;
  detail: string;
}

export interface Prediction {
  top_condition: string;
  confidence: number;
  severity: string;
  alternatives: { disease: string; probability: number }[];
  reasoning_summary: string;
  evidence_sources: string[];
  evidence: EvidenceChunk[];
  explanation?: Explanation | null;
  ayurvedic_guidance?: AyurvedicGuidance;
  triage?: Triage;
  clinical_context: Record<string, string>;
  denied_symptoms: string[];
  mentioned_conditions: { condition: string; attribution: string; status: string; text: string }[];
  watch_for: string[];
  /** True when the evidence did not support naming one condition. */
  abstained?: boolean;
  abstain_reason?: string | null;
  disclaimer: string;
}

export interface StatsResponse {
  model_type: string;
  diseases: number;
  symptoms: number;
  kb_documents: number;
  kb_chunks: number;
  embedding_model: string;
  red_flag_patterns: number;
  ayurveda_entries: number;
  llm_mode: string;
}

export interface ReportResponse {
  session_id: string;
  generated_at: string;
  duration_turns: number;
  symptoms: string[];
  clinical_context: Record<string, string>;
  prediction?: Record<string, any>;
  transcript: { role: string; content: string; created_at: string }[];
  markdown: string;
}

export interface AyurvedicGuidance {
  dosha_imbalance: string;
  diet: string;
  lifestyle: string;
  general_herbs_note: string;
  caution: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: { role: string; content: string; metadata_?: Record<string, unknown>; created_at: string }[];
  state: string;
  accumulated_symptoms: Record<string, boolean>;
}

export const api = {
  createSession: () => request<{ session_id: string }>('/api/session', { method: 'POST' }),

  sendChat: (session_id: string, message: string, profile?: unknown) =>
    request<ChatResponse>('/api/chat', {
      method: 'POST',
      // The profile travels with the turn and is never stored server-side.
      body: JSON.stringify({ session_id, message, profile }),
    }),

  getHistory: (session_id: string) =>
    request<ChatHistoryResponse>(`/api/chat/history/${session_id}`),

  health: () => request<{ status: string; model_loaded: boolean; vector_store_ready: boolean }>('/api/health'),

  stats: () => request<StatsResponse>('/api/stats'),

  safetyCategories: () =>
    request<{ categories: { key: string; description: string }[] }>(
      '/api/safety/categories',
    ),

  report: (session_id: string) => request<ReportResponse>(`/api/session/${session_id}/report`),

  analyzeText: (text: string, use_semantic = false) =>
    request<NlpAnalysis>('/api/nlp/analyze', {
      method: 'POST',
      body: JSON.stringify({ text, use_semantic }),
    }),

  nlpCapabilities: () => request<NlpCapabilities>('/api/nlp/capabilities'),

  evaluation: () => request<EvaluationResponse>('/api/evaluation'),
};

export type MessageState = 'greeting' | 'symptom_collection' | 'clarifying' | 'result' | 'emergency';

// ---------------------------------------------------------------------------
// "I know my condition" mode
// ---------------------------------------------------------------------------

export interface ConditionSummary {
  id: string;
  name: string;
  urgency: string;
  is_contagious: boolean;
}

export interface ConditionFact {
  id: string;
  name: string;
  aliases: string[];
  urgency: string;
  severity_tier: string;
  overview: string;
  common_symptoms: string[];
  causes: string;
  risk_factors: string[];
  is_contagious: boolean;
  typical_course: string;
  self_care: string;
  diet: { eat: string; avoid: string };
  ayurvedic_lifestyle: string;
  ayurvedic_caution: string;
  complications: string[];
  condition_specific_red_flags: string[];
  when_to_see_doctor: string;
  questions_to_ask_doctor: string[];
  common_tests: string[];
  sources: string[];
}

export interface ConditionMatchResponse {
  status: 'matched' | 'ambiguous' | 'not_found' | 'emergency';
  condition?: ConditionFact;
  candidates?: ConditionSummary[];
  confidence?: number;
  strategy?: string;
  message?: string;
  chips?: string[];
  is_emergency_condition?: boolean;
  emergency_note?: string | null;
  disclaimer: string;
}

export interface ConditionAnswer {
  status: 'ok' | 'emergency' | 'not_found';
  condition?: ConditionSummary;
  answer: string;
  intent: string;
  confidence: number;
  intent_source: string;
  sections_used: string[];
  passages?: { id: string; source: string; text: string }[];
  chips?: string[];
  message?: string;
  disclaimer: string;
}

export const conditionApi = {
  list: () => request<{ conditions: ConditionSummary[] }>('/api/condition/list'),

  get: (id: string) =>
    request<{
      found: boolean;
      condition?: ConditionFact;
      chips?: string[];
      is_emergency_condition?: boolean;
      emergency_note?: string | null;
      disclaimer: string;
    }>(`/api/condition/${id}`),

  match: (text: string) =>
    request<ConditionMatchResponse>('/api/condition/match', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),

  ask: (condition_id: string, question: string, intent?: string) =>
    request<ConditionAnswer>('/api/condition/ask', {
      method: 'POST',
      body: JSON.stringify({ condition_id, question, intent }),
    }),
};