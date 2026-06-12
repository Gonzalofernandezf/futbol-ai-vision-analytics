export interface EvalHistoryEntry {
  timestamp: string;
  git_hash: string;
  git_message: string;
  git_date: string;
  model: string;
  conf_threshold: number;
  n_frames: number;
  overall_pck_5px: number;
  overall_pck_10px: number;
  overall_pck_20px: number;
  mean_error_px: number | null;
  mean_homography_error_m: number | null;
  homography_failure_rate: number;
  mean_kps_detected_per_frame: number;
}

export interface EvalPerKeypoint {
  n_gt_visible: number;
  n_detected: number;
  recall: number;
  mean_error_px: number | null;
  pck_5px: number;
  pck_10px: number;
  pck_20px: number;
}

export interface EvalPerFrame {
  filename: string;
  n_gt_visible: number;
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
    overall_pck_5px: number;
    overall_pck_10px: number;
    overall_pck_20px: number;
    mean_error_px: number | null;
    median_error_px: number | null;
    homography_failure_rate: number;
    mean_homography_error_m: number | null;
    mean_kps_detected_per_frame: number;
    mean_kps_gt_per_frame: number;
  };
  per_keypoint: Record<string, EvalPerKeypoint>;
  per_frame: EvalPerFrame[];
}
