export interface ProcessingMeta {
  match_id: string;
  processed_at: string;
  pipeline_version: string;
  models: {
    players: string;
    ball: string;
    field: string;
  };
  timing: {
    total_seconds: number;
    chunks_processed: number;
    per_chunk_avg_seconds: number;
    slowest_chunk_seconds: number;
    fastest_chunk_seconds: number;
    phases: Record<string, number>;
  };
  sanity_checks: Array<{
    label: string;
    value: string;
    status: "ok" | "warn" | "error";
    expected: string;
  }>;
  infra: {
    gpu_model: string;
    gpu_util_avg_pct: number;
    estimated_cost_usd: number;
  };
}