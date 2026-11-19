export type DisplayMode = 'pbr' | 'wireframe' | 'clay';

export type JobStage =
  | 'queued'
  | 'preprocess'
  | 'inference'
  | 'mesh'
  | 'export'
  | 'done'
  | 'error';

export type JobState = 'queued' | 'running' | 'completed' | 'failed';

export interface ReconstructResult {
  job_id: string;
  status: 'completed';
  glb_url: string;
  download_url: string;
  file_name: string;
  elapsed_seconds: number;
  preprocess_seconds: number;
  inference_seconds: number;
  mesh_seconds: number;
  vertices: number;
  faces: number;
  texture: string;
  input_views: number;
  selected_view: number;
  device: string;
}

export interface JobStatus {
  job_id: string;
  status: JobState;
  stage: JobStage;
  progress: number;
  message: string;
  created_at: number;
  updated_at: number;
  input_views: number;
  result: ReconstructResult | null;
  error: string | null;
}

export interface HealthInfo {
  status: string;
  app: string;
  version: string;
  device: string;
  gpu: string | null;
  vram_total_mb: number | null;
  vram_allocated_mb: number | null;
  model_id: string;
  model_loaded: boolean;
  queue_depth: number;
  uptime_seconds: number;
  max_images: number;
  max_upload_mb: number;
}

export interface ReconstructOptions {
  foregroundRatio: number;
  mcResolution: number;
  targetFaces: number;
  removeBackground: boolean;
}

export interface ReconstructHandlers {
  onUploadProgress?: (ratio: number) => void;
  onStage?: (stage: JobStage, progress: number, message: string) => void;
}

export interface ModelStats {
  vertices: number;
  triangles: number;
}

export interface UploadItem {
  id: string;
  file: File;
  url: string;
  width: number;
  height: number;
}

export interface SlicedView {
  file: File;
  row: number;
  column: number;
  index: number;
}
