export enum AgentStatus {
  IDLE = 'IDLE',
  WORKING = 'WORKING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR'
}

export interface DataPoint {
  id: number;
  [key: string]: number | string;
}

// --- MASTER SPEC: SECTION 1 (Output Schema) ---

export interface ColumnSchema {
  dtype: string;
  role: 'numeric' | 'categorical' | 'datetime' | 'index' | 'unknown';
}

export interface NumericStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p01: number;
  p25: number;
  p50: number;
  p75: number;
  p99: number;
  skew: number;
  kurtosis: number;
}

export interface HistogramBin {
  range: string;
  count: number;
  min: number;
  max: number;
}

export interface NumericProfile {
  stats: NumericStats;
  histogram: HistogramBin[];
  missing: { count: number; fraction: number };
  zeros: { count: number; fraction: number };
  outliers: { count: number; fraction: number };
}

export interface CategoricalProfile {
  n_unique: number;
  top_values: { value: string; count: number; fraction: number }[];
  other_fraction: number;
  missing: { count: number; fraction: number };
}

export interface CorrelationPair {
  pair: [string, string];
  pearson: number;
  n_samples: number;
}

export interface TimeMetricProfile {
  trend: { slope: number; r2: number; direction: 'up' | 'down' | 'flat' };
  change_points: { index: number; value: number; delta_mean: number }[];
}

export interface TimeProfile {
  index_column: string;
  index_range: { min: number | string; max: number | string };
  metrics: Record<string, TimeMetricProfile>;
}

export interface GlobalAnomalies {
  total_flagged_rows: number;
  fraction: number;
  rules: { rule: string; count: number }[];
}

export interface DataSummary {
  fileName: string;
  n_rows: number;
  n_cols: number;
  schema: Record<string, ColumnSchema>;
  numeric_profile: Record<string, NumericProfile>;
  categorical_profile: Record<string, CategoricalProfile>;
  top_correlations: CorrelationPair[];
  time_profiles?: TimeProfile;
  anomalies: GlobalAnomalies;
  sample_rows: DataPoint[]; // Capped at 5 rows for context
  raw_data_subset?: DataPoint[]; // WARNING: For visualization ONLY. NEVER send this to the LLM.
}

export interface AgentState {
  id: string;
  name: string;
  role: string;
  status: AgentStatus;
  message: string;
}

export interface AnalysisReport {
  technicalReport: string;
  executiveSummary: string;
  anomalies: string[];
}