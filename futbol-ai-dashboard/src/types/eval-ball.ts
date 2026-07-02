export interface EvalBallHistoryEntry {
  timestamp: string;
  git_hash: string | null;
  git_message: string | null;
  git_date: string | null;
  model: string;
  n_frames: number;
  n_gt_positive: number | null;
  n_gt_negative: number | null;
  recall: number | null;
  precision: number | null;
  f1: number | null;
  overall_pck_5px: number | null;
  overall_pck_10px: number | null;
  overall_pck_20px: number | null;
  mean_error_px: number | null;
  false_positive_rate: number | null;
}

export interface EvalBallPerFrame {
  filename: string;
  gt_present: boolean;
  pred_present: boolean;
  outcome: "TP" | "FN" | "FP" | "TN";
  error_px: number | null;
  confidence: number | null;
}

export interface EvalBallReport {
  metadata: {
    timestamp: string;
    n_frames: number;
    n_gt_positive: number;
    n_gt_negative: number;
    model: string;
    model_task: string;
    pck_thresholds_px: number[];
    thresholds: {
      yolo_ball_conf: number;
      ball_min_conf: number;
      bbox_px: [number, number];
      aspect: [number, number];
    };
    git_commit: {
      hash: string | null;
      short_hash: string | null;
      message: string | null;
      date: string | null;
    };
  };
  summary: {
    recall: number | null;
    precision: number | null;
    f1: number | null;
    false_positive_rate: number | null;
    overall_pck_5px: number | null;
    overall_pck_10px: number | null;
    overall_pck_20px: number | null;
    mean_error_px: number | null;
    median_error_px: number | null;
    mean_confidence_tp: number | null;
  };
  per_frame: EvalBallPerFrame[];
}
