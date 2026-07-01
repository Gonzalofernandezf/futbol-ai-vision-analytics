export interface EvalHistoryEntry {
  timestamp: string;
  git_hash: string | null;
  git_message: string | null;
  git_date: string | null;
  model: string;
  conf_threshold: number;
  n_frames: number;
  overall_recall: number | null;
  overall_pck_5px: number | null;
  overall_pck_10px: number | null;
  overall_pck_20px: number | null;
  mean_error_px: number | null;
  mean_homography_error_m: number | null;
  homography_failure_rate: number | null;
  mean_kps_detected_per_frame: number;
}

export interface EvalPerKeypoint {
  class_id: number;
  n_gt: number;
  n_detected: number;
  recall: number | null;
  mean_error_px: number | null;
  pck_5px: number | null;
  pck_10px: number | null;
  pck_20px: number | null;
}

export interface EvalPerFrame {
  filename: string;
  n_gt: number;
  n_detected: number;
  n_matched: number;
  mean_error_px: number | null;
  homography_ok: boolean;
  homography_error_m: number | null;
}

export interface EvalReport {
  metadata: {
    timestamp: string;
    n_frames: number;
    model: string;
    conf_threshold: number;
    pck_thresholds_px: number[];
    git_commit: {
      hash: string;
      short_hash: string;
      message: string;
      date: string;
    };
  };
  summary: {
    overall_recall: number | null;
    overall_pck_5px: number | null;
    overall_pck_10px: number | null;
    overall_pck_20px: number | null;
    mean_error_px: number | null;
    median_error_px: number | null;
    homography_failure_rate: number | null;
    mean_homography_error_m: number | null;
    mean_kps_detected_per_frame: number;
    mean_kps_gt_per_frame: number;
  };
  per_keypoint: Record<string, EvalPerKeypoint>;
  per_frame: EvalPerFrame[];
}
