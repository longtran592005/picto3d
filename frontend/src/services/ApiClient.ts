import type {
  HealthInfo,
  JobStage,
  JobStatus,
  ReconstructHandlers,
  ReconstructOptions,
  ReconstructResult,
} from '../types';

const POLL_TIMEOUT_MS = 30 * 60 * 1000;
const UPLOAD_TIMEOUT_MS = 20 * 60 * 1000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function normalizeBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) {
    throw new ApiError('Chưa cấu hình Backend API URL.');
  }
  if (!/^https?:\/\//i.test(trimmed)) {
    const isLocal = /^(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$)/i.test(trimmed);
    return `${isLocal ? 'http' : 'https'}://${trimmed}`;
  }
  return trimmed;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isReconstructResult(value: unknown): value is ReconstructResult {
  if (!isRecord(value)) return false;
  return (
    isString(value.job_id) &&
    isString(value.glb_url) &&
    isString(value.download_url) &&
    isString(value.file_name) &&
    isNumber(value.elapsed_seconds) &&
    isNumber(value.vertices) &&
    isNumber(value.faces) &&
    isNumber(value.input_views) &&
    isNumber(value.selected_view)
  );
}

export function isJobStatus(value: unknown): value is JobStatus {
  if (!isRecord(value)) return false;
  const validStage =
    isString(value.stage) &&
    ['queued', 'preprocess', 'inference', 'mesh', 'export', 'done', 'error'].includes(
      value.stage,
    );
  const validState =
    isString(value.status) &&
    ['queued', 'running', 'completed', 'failed'].includes(value.status);
  if (!validState || !validStage) return false;
  if (value.result !== null && value.result !== undefined && !isReconstructResult(value.result)) {
    return false;
  }
  return isString(value.job_id) && isNumber(value.progress);
}

function isHealthInfo(value: unknown): value is HealthInfo {
  return (
    isRecord(value) &&
    isString(value.app) &&
    isString(value.device) &&
    typeof value.model_loaded === 'boolean' &&
    isNumber(value.queue_depth)
  );
}

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function extractDetail(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const detail = value.detail;
  if (isString(detail)) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry) => {
        if (isRecord(entry) && isString(entry.msg)) {
          const location = Array.isArray(entry.loc) ? entry.loc.join('.') : '';
          return location ? `${location}: ${entry.msg}` : entry.msg;
        }
        return null;
      })
      .filter((entry): entry is string => entry !== null);
    if (parts.length > 0) return parts.join(' · ');
  }
  if (isRecord(value) && isString(value.message)) return value.message;
  return null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function dispatchDownload(url: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  setBaseUrl(baseUrl: string): void {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  private url(path: string): string {
    return new URL(path, `${this.baseUrl}/`).toString();
  }

  async health(): Promise<HealthInfo> {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 9000);
    try {
      const response = await fetch(this.url('/api/health'), {
        method: 'GET',
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: controller.signal,
      });
      const data: unknown = parseJson(await response.text());
      if (!response.ok) {
        throw new ApiError(
          extractDetail(data) ?? `Backend trả HTTP ${response.status}.`,
          response.status,
        );
      }
      if (!isHealthInfo(data)) {
        throw new ApiError('Phản hồi /api/health không đúng định dạng.');
      }
      return data;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new ApiError('Backend không phản hồi (timeout 9s).');
      }
      throw new ApiError(
        'Không kết nối được backend. Kiểm tra URL, tunnel và CORS.',
      );
    } finally {
      window.clearTimeout(timer);
    }
  }

  private upload<T>(form: FormData, handlers: ReconstructHandlers): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.url('/api/reconstruct?wait=false'));
      xhr.responseType = 'text';
      xhr.timeout = UPLOAD_TIMEOUT_MS;
      xhr.setRequestHeader('Accept', 'application/json');

      xhr.upload.onprogress = (event: ProgressEvent) => {
        if (event.lengthComputable && event.total > 0 && handlers.onUploadProgress) {
          handlers.onUploadProgress(Math.min(1, event.loaded / event.total));
        }
      };

      xhr.onload = () => {
        const data: unknown = parseJson(xhr.responseText ?? '');
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(data as T);
          return;
        }
        const detail = extractDetail(data);
        const fallback =
          xhr.status === 413
            ? 'Ảnh quá lớn so với giới hạn của backend.'
            : xhr.status === 0
              ? 'Mất kết nối tới backend.'
              : `Backend trả HTTP ${xhr.status}.`;
        reject(new ApiError(detail ?? fallback, xhr.status, detail ?? undefined));
      };

      xhr.onerror = () => {
        reject(
          new ApiError(
            'Không gửi được ảnh. Kiểm tra tunnel, CORS và kết nối mạng.',
          ),
        );
      };

      xhr.ontimeout = () => {
        reject(new ApiError('Upload quá thời gian chờ (20 phút).'));
      };

      xhr.send(form);
    });
  }

  async reconstruct(
    files: File[],
    options: ReconstructOptions,
    handlers: ReconstructHandlers = {},
  ): Promise<ReconstructResult> {
    if (files.length === 0) {
      throw new ApiError('Chưa có ảnh để dựng mô hình.');
    }

    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }
    form.append('foreground_ratio', options.foregroundRatio.toFixed(2));
    form.append('mc_resolution', String(options.mcResolution));
    form.append('target_faces', String(options.targetFaces));
    form.append('remove_background', String(options.removeBackground));

    const accepted = await this.upload<unknown>(form, handlers);

    if (isReconstructResult(accepted)) {
      handlers.onStage?.('done', 1, 'Hoàn tất');
      return accepted;
    }

    if (isJobStatus(accepted)) {
      handlers.onStage?.(accepted.stage, accepted.progress, accepted.message);
      return this.pollJob(accepted.job_id, handlers);
    }

    throw new ApiError('Backend trả dữ liệu không hợp lệ.');
  }

  private async pollJob(
    jobId: string,
    handlers: ReconstructHandlers,
  ): Promise<ReconstructResult> {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    let interval = 550;

    while (Date.now() < deadline) {
      await sleep(interval);
      interval = Math.min(1200, interval + 60);

      let response: Response;
      try {
        response = await fetch(this.url(`/api/jobs/${encodeURIComponent(jobId)}`), {
          headers: { Accept: 'application/json' },
          cache: 'no-store',
          signal: AbortSignal.timeout(15_000),
        });
      } catch {
        throw new ApiError('Mất kết nối khi đang chờ kết quả từ backend.');
      }

      const data: unknown = parseJson(await response.text());
      if (!response.ok) {
        throw new ApiError(
          extractDetail(data) ?? `Không đọc được trạng thái job (${response.status}).`,
          response.status,
        );
      }
      if (!isJobStatus(data)) {
        throw new ApiError('Trạng thái job không đúng định dạng.');
      }

      handlers.onStage?.(data.stage, data.progress, data.message);

      if (data.status === 'completed') {
        if (data.result) return data.result;
        throw new ApiError('Job hoàn tất nhưng thiếu thông tin mô hình.');
      }
      if (data.status === 'failed') {
        throw new ApiError(data.error ?? 'Backend không dựng được mô hình.');
      }
    }

    throw new ApiError('Quá thời gian chờ mô hình (30 phút).');
  }

  async download(url: string, filename: string): Promise<void> {
    try {
      const response = await fetch(url, {
        mode: 'cors',
        cache: 'no-store',
        signal: AbortSignal.timeout(120_000),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      dispatchDownload(objectUrl, filename);
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 2000);
    } catch {
      // Fallback: the backend serves Content-Disposition: attachment, so a
      // direct navigation downloads the file even without CORS access.
      dispatchDownload(url, filename);
    }
  }
}

export type { JobStage };
