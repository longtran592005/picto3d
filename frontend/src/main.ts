import './styles.css';

import { StudioScene } from './core/StudioScene';
import { ApiClient, ApiError } from './services/ApiClient';
import type {
  DisplayMode,
  HealthInfo,
  JobStage,
  ModelStats,
  ReconstructResult,
  UploadItem,
} from './types';
import {
  allCells,
  clampGridValue,
  drawGridPreview,
  gridIndex,
  loadImageBitmap,
  readImageSize,
  sliceImage,
} from './utils/ImageSlicer';

const API_STORAGE_KEY = 'picto3d.apiUrl';
const DEFAULT_API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const MAX_UPLOAD_MB = 25;
const MAX_IMAGES = 8;
const ALLOWED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const STAGE_ORDER: JobStage[] = ['preprocess', 'inference', 'mesh', 'export'];
const STAGE_LABELS: Record<JobStage, string> = {
  queued: 'Đang xếp hàng…',
  preprocess: 'Đang tách nền…',
  inference: 'Đang dựng hình 3D…',
  mesh: 'Đang tối ưu lưới…',
  export: 'Đang xuất GLB…',
  done: 'Hoàn tất',
  error: 'Xử lý thất bại',
};

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Thiếu phần tử #${id} trong giao diện.`);
  }
  return element as T;
}

const stageEl = byId<HTMLElement>('stage');
const canvasEl = byId<HTMLCanvasElement>('viewer-canvas');
const apiUrlInput = byId<HTMLInputElement>('api-url');
const apiCheckBtn = byId<HTMLButtonElement>('api-check');
const apiStatusDot = byId<HTMLElement>('api-status-dot');
const apiStatusText = byId<HTMLElement>('api-status-text');
const dropzone = byId<HTMLElement>('dropzone');
const fileInput = byId<HTMLInputElement>('file-input');
const thumbsEl = byId<HTMLElement>('thumbs');
const viewCountEl = byId<HTMLElement>('view-count');
const sliceToggle = byId<HTMLInputElement>('slice-toggle');
const slicePanel = byId<HTMLElement>('slice-panel');
const sliceCanvas = byId<HTMLCanvasElement>('slice-canvas');
const sliceCountEl = byId<HTMLElement>('slice-count');
const gridRowsInput = byId<HTMLInputElement>('grid-rows');
const gridColsInput = byId<HTMLInputElement>('grid-cols');
const fgRatioInput = byId<HTMLInputElement>('fg-ratio');
const fgRatioValue = byId<HTMLOutputElement>('fg-ratio-value');
const qualitySelect = byId<HTMLSelectElement>('quality');
const targetFacesSelect = byId<HTMLSelectElement>('target-faces');
const removeBgInput = byId<HTMLInputElement>('remove-bg');
const generateBtn = byId<HTMLButtonElement>('generate');
const generateLabel = generateBtn.querySelector<HTMLElement>('.btn-label');
const generateSpinner = generateBtn.querySelector<HTMLElement>('.btn-spinner');
const progressEl = byId<HTMLElement>('progress');
const progressStage = byId<HTMLElement>('progress-stage');
const progressPercent = byId<HTMLElement>('progress-percent');
const progressFill = byId<HTMLElement>('progress-fill');
const progressSteps = byId<HTMLOListElement>('progress-steps');
const errorBox = byId<HTMLElement>('error-box');
const resultCard = byId<HTMLElement>('result-card');
const resultTime = byId<HTMLElement>('result-time');
const statVertices = byId<HTMLElement>('stat-vertices');
const statFaces = byId<HTMLElement>('stat-faces');
const downloadBtn = byId<HTMLButtonElement>('download-btn');
const displayModes = byId<HTMLElement>('display-modes');
const envIntensity = byId<HTMLInputElement>('env-intensity');
const autoRotateBtn = byId<HTMLButtonElement>('autorotate-btn');
const resetBtn = byId<HTMLButtonElement>('reset-btn');
const emptyState = byId<HTMLElement>('empty-state');
const demoBtn = byId<HTMLButtonElement>('demo-btn');
const viewerLoader = byId<HTMLElement>('viewer-loader');
const viewerLoaderText = byId<HTMLElement>('viewer-loader-text');
const stageHint = byId<HTMLElement>('stage-hint');
const modelStatsEl = byId<HTMLElement>('model-stats');
const toastEl = byId<HTMLElement>('toast');

const studio = new StudioScene(stageEl, canvasEl);
const numberFormatter = new Intl.NumberFormat('vi-VN');

let uploads: UploadItem[] = [];
let sliceEnabled = false;
let sliceBitmap: ImageBitmap | null = null;
let sliceLoadToken = 0;
let gridRows = 2;
let gridColumns = 2;
let selectedCells = new Set<number>();
let progressValue = 0;
let busy = false;
let healthInfo: HealthInfo | null = null;
let currentResult: ReconstructResult | null = null;
let downloadClient: ApiClient | null = null;
let modelLoadToken = 0;
let toastTimer = 0;
let resizeTimer = 0;

function formatNumber(value: number): string {
  return numberFormatter.format(Math.max(0, Math.round(value)));
}

function showToast(message: string, kind: 'info' | 'success' | 'error' = 'info'): void {
  window.clearTimeout(toastTimer);
  toastEl.textContent = message;
  toastEl.className = `toast ${kind}`;
  toastEl.hidden = false;
  toastTimer = window.setTimeout(() => {
    toastEl.hidden = true;
  }, kind === 'error' ? 6000 : 4000);
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'Đã có lỗi không xác định.';
}

function setApiStatus(state: 'idle' | 'checking' | 'online' | 'offline', text: string): void {
  apiStatusDot.dataset.state = state;
  apiStatusText.textContent = text;
  apiStatusText.classList.toggle('positive', state === 'online');
  apiStatusText.classList.toggle('negative', state === 'offline');
}

function readStoredApiUrl(): string {
  try {
    return window.localStorage.getItem(API_STORAGE_KEY) ?? DEFAULT_API_URL;
  } catch {
    return DEFAULT_API_URL;
  }
}

function storeApiUrl(value: string): void {
  try {
    window.localStorage.setItem(API_STORAGE_KEY, value);
  } catch {
    // Storage can be unavailable in private mode; the session still works.
  }
}

function getClient(): ApiClient {
  const value = apiUrlInput.value.trim();
  if (!value) {
    throw new ApiError('Nhập Backend API URL trước khi dùng.');
  }
  return new ApiClient(value);
}

async function checkHealth(notify = false): Promise<void> {
  setApiStatus('checking', 'Đang kiểm tra kết nối…');
  try {
    const client = getClient();
    const info = await client.health();
    healthInfo = info;
    const gpu = info.gpu
      ? `${info.gpu}${info.vram_total_mb ? ` · ${Math.round(info.vram_total_mb / 1024)} GB` : ''}`
      : info.device.toUpperCase();
    const model = info.model_loaded ? 'model sẵn sàng' : 'model đang nạp';
    setApiStatus('online', `${gpu} · ${model}`);
    if (notify) showToast('Kết nối backend thành công.', 'success');
  } catch (error) {
    healthInfo = null;
    setApiStatus('offline', errorMessage(error));
  }
}

// ------------------------------------------------------------------- progress
function setProgressUi(
  value: number,
  label: string,
  stage: JobStage | 'upload' | 'loading',
): void {
  const clamped = Math.max(progressValue, Math.min(1, Math.max(0, value)));
  progressValue = clamped;
  progressFill.style.width = `${Math.round(clamped * 100)}%`;
  progressPercent.textContent = `${Math.round(clamped * 100)}%`;
  progressStage.textContent = label;
  updateSteps(stage);
}

function updateSteps(stage: JobStage | 'upload' | 'loading'): void {
  const effective: JobStage =
    stage === 'upload' ? 'preprocess' : stage === 'loading' ? 'export' : stage;
  const activeIndex = effective === 'done' ? STAGE_ORDER.length : STAGE_ORDER.indexOf(effective);
  const items = progressSteps.querySelectorAll<HTMLLIElement>('li[data-step]');
  items.forEach((item) => {
    const index = STAGE_ORDER.indexOf(item.dataset.step as JobStage);
    item.classList.toggle('done', activeIndex > index || effective === 'done');
    item.classList.toggle('active', activeIndex === index && effective !== 'done');
  });
}

function resetProgress(): void {
  progressValue = 0;
  progressFill.style.width = '0%';
  progressPercent.textContent = '0%';
  progressStage.textContent = 'Đang chuẩn bị…';
  for (const item of progressSteps.querySelectorAll('li')) {
    item.classList.remove('active', 'done');
  }
  progressEl.hidden = false;
}

function showError(message: string): void {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError(): void {
  errorBox.textContent = '';
  errorBox.hidden = true;
}

function setBusy(value: boolean): void {
  busy = value;
  generateBtn.classList.toggle('busy', value);
  if (generateLabel) generateLabel.textContent = value ? 'Đang xử lý…' : 'Tạo mô hình 3D';
  if (generateSpinner) generateSpinner.hidden = !value;
  updateGenerateState();
}

function updateGenerateState(): void {
  const needsSlices = sliceEnabled && selectedCells.size === 0;
  generateBtn.disabled = busy || uploads.length === 0 || needsSlices;
}

// -------------------------------------------------------------------- uploads
function fileNameOf(index: number): string {
  return `Góc nhìn ${index + 1}`;
}

function renderUploads(): void {
  viewCountEl.textContent = String(uploads.length);
  thumbsEl.hidden = uploads.length === 0;
  thumbsEl.replaceChildren();

  uploads.forEach((item, index) => {
    const thumb = document.createElement('div');
    thumb.className = 'thumb';

    const image = document.createElement('img');
    image.src = item.url;
    image.alt = fileNameOf(index);
    image.loading = 'lazy';

    const badge = document.createElement('span');
    badge.className = 'thumb-index';
    badge.textContent = String(index + 1);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'thumb-remove';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Xoá ${fileNameOf(index)}`);
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      removeUpload(item.id);
    });

    thumb.append(image, badge, remove);
    thumbsEl.append(thumb);
  });

  updateGenerateState();
}

function createUploadId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `img-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function maxImagesAllowed(): number {
  return healthInfo?.max_images ?? MAX_IMAGES;
}

function maxUploadMb(): number {
  return healthInfo?.max_upload_mb ?? MAX_UPLOAD_MB;
}

async function addFiles(fileList: FileList | File[]): Promise<void> {
  const incoming = Array.from(fileList);
  if (incoming.length === 0) return;

  const limit = maxImagesAllowed();
  if (sliceEnabled && uploads.length + incoming.length > 1) {
    disableSlicer();
    showToast('Grid Slicer chỉ áp dụng cho 1 ảnh ghép — đã tắt.', 'info');
  }

  if (uploads.length >= limit) {
    showToast(`Tối đa ${limit} ảnh mỗi lần dựng.`, 'error');
    return;
  }

  for (const file of incoming) {
    if (uploads.length >= limit) {
      showToast(`Chỉ nhận tối đa ${limit} ảnh, các ảnh thừa đã bỏ qua.`, 'error');
      break;
    }
    if (!ALLOWED_TYPES.has(file.type)) {
      showToast(`"${file.name}" không phải PNG, JPG hoặc WEBP.`, 'error');
      continue;
    }
    if (file.size > maxUploadMb() * 1024 * 1024) {
      showToast(`"${file.name}" vượt quá ${maxUploadMb()} MB.`, 'error');
      continue;
    }

    const url = URL.createObjectURL(file);
    try {
      const size = await readImageSize(url);
      uploads.push({
        id: createUploadId(),
        file,
        url,
        width: size.width,
        height: size.height,
      });
    } catch {
      URL.revokeObjectURL(url);
      showToast(`Không đọc được ảnh "${file.name}".`, 'error');
    }
  }

  renderUploads();
  if (sliceEnabled) {
    void refreshSlicer();
  }
}

function removeUpload(id: string): void {
  const item = uploads.find((entry) => entry.id === id);
  if (!item) return;
  URL.revokeObjectURL(item.url);
  uploads = uploads.filter((entry) => entry.id !== id);

  if (sliceEnabled && uploads.length !== 1) {
    disableSlicer();
  }

  renderUploads();
  if (sliceEnabled) {
    void refreshSlicer();
  }
}

// --------------------------------------------------------------------- slicer
function clearSlicePreview(): void {
  const context = sliceCanvas.getContext('2d');
  if (context) {
    context.clearRect(0, 0, sliceCanvas.width, sliceCanvas.height);
  }
}

function updateSliceCount(): void {
  sliceCountEl.textContent = `${selectedCells.size}/${gridRows * gridColumns} ô`;
}

function drawSlicer(): void {
  if (!sliceEnabled || !sliceBitmap) return;
  drawGridPreview(sliceCanvas, sliceBitmap, {
    rows: gridRows,
    columns: gridColumns,
    selected: selectedCells,
  });
  updateSliceCount();
}

async function refreshSlicer(): Promise<void> {
  const source = uploads[0];
  sliceLoadToken += 1;
  const token = sliceLoadToken;

  if (!source) {
    sliceBitmap?.close();
    sliceBitmap = null;
    clearSlicePreview();
    updateSliceCount();
    updateGenerateState();
    return;
  }

  try {
    const bitmap = await loadImageBitmap(source.file);
    if (token !== sliceLoadToken) {
      bitmap.close();
      return;
    }
    sliceBitmap?.close();
    sliceBitmap = bitmap;
    selectedCells = allCells(gridRows, gridColumns);
    drawSlicer();
    updateGenerateState();
  } catch (error) {
    showToast(errorMessage(error), 'error');
    disableSlicer();
  }
}

function enableSlicer(): void {
  if (uploads.length !== 1) {
    sliceToggle.checked = false;
    sliceEnabled = false;
    slicePanel.hidden = true;
    showToast('Hãy chọn đúng 1 ảnh ghép nhiều góc để dùng Grid Slicer.', 'info');
    return;
  }

  sliceEnabled = true;
  slicePanel.hidden = false;
  selectedCells = allCells(gridRows, gridColumns);
  updateGenerateState();
  void refreshSlicer();
}

function disableSlicer(): void {
  sliceEnabled = false;
  sliceToggle.checked = false;
  slicePanel.hidden = true;
  sliceBitmap?.close();
  sliceBitmap = null;
  selectedCells = new Set<number>();
  updateSliceCount();
  clearSlicePreview();
  updateGenerateState();
}

function applyGridSize(): void {
  gridRows = clampGridValue(Number(gridRowsInput.value), 2);
  gridColumns = clampGridValue(Number(gridColsInput.value), 2);
  gridRowsInput.value = String(gridRows);
  gridColsInput.value = String(gridColumns);
  selectedCells = allCells(gridRows, gridColumns);
  drawSlicer();
  updateGenerateState();
}

// -------------------------------------------------------------------- viewer
function updateModelMeta(stats: ModelStats): void {
  modelStatsEl.textContent = `${formatNumber(stats.vertices)} đỉnh · ${formatNumber(stats.triangles)} mặt`;
  modelStatsEl.hidden = false;
}

async function showModel(url: string, result: ReconstructResult | null): Promise<boolean> {
  const token = ++modelLoadToken;
  viewerLoader.hidden = false;
  viewerLoaderText.textContent = 'Đang tải mô hình…';
  emptyState.hidden = true;

  try {
    const stats = await studio.loadFromUrl(url, (ratio) => {
      if (token === modelLoadToken) {
        viewerLoaderText.textContent = `Đang tải mô hình… ${Math.round(ratio * 100)}%`;
      }
    });

    if (token !== modelLoadToken) return false;

    updateModelMeta(stats);
    if (result) {
      resultCard.hidden = false;
      resultTime.textContent = `${result.elapsed_seconds.toFixed(1)}s`;
      statVertices.textContent = formatNumber(result.vertices);
      statFaces.textContent = formatNumber(result.faces);
      stageHint.textContent = `View ${result.selected_view}/${result.input_views} · ${result.device.toUpperCase()} · ${result.texture}`;
    }
    return true;
  } catch (error) {
    if (token !== modelLoadToken) return false;
    if (!studio.getStats().vertices) {
      emptyState.hidden = false;
    }
    showError(`Mô hình đã tạo nhưng chưa tải được vào viewer: ${errorMessage(error)}`);
    return false;
  } finally {
    if (token === modelLoadToken) {
      viewerLoader.hidden = true;
    }
  }
}

async function loadDemo(): Promise<void> {
  if (busy) return;
  const url = `${import.meta.env.BASE_URL}demo/damaged-helmet.glb`;
  currentResult = null;
  resultCard.hidden = true;
  downloadClient = null;
  showError('');
  clearError();
  stageHint.textContent = 'Demo · DamagedHelmet © theblueturtle_/ctxwing · CC BY-NC 4.0';
  const ok = await showModel(url, null);
  if (!ok) {
    stageHint.textContent = 'Kéo để xoay · Lăn chuột để zoom · Chuột phải để dịch chuyển';
  }
}

// ------------------------------------------------------------------ generate
function currentStageLabel(stage: JobStage, message: string): string {
  return message.trim() || STAGE_LABELS[stage];
}

async function collectFiles(): Promise<File[]> {
  if (!sliceEnabled) {
    return uploads.map((item) => item.file);
  }
  if (!sliceBitmap) {
    throw new Error('Chưa tải được ảnh ghép để cắt lưới.');
  }
  const views = await sliceImage(
    sliceBitmap,
    { rows: gridRows, columns: gridColumns, selected: new Set(selectedCells) },
    'view',
  );
  if (views.length === 0) {
    throw new Error('Hãy chọn ít nhất một góc nhìn trong lưới.');
  }
  return views.map((view) => view.file);
}

async function generate(): Promise<void> {
  if (busy) return;

  let client: ApiClient;
  try {
    client = getClient();
  } catch (error) {
    showToast(errorMessage(error), 'error');
    apiUrlInput.focus();
    return;
  }

  storeApiUrl(apiUrlInput.value.trim());
  downloadClient = client;
  currentResult = null;
  resultCard.hidden = true;
  clearError();
  setBusy(true);
  resetProgress();
  setProgressUi(0.02, 'Đang chuẩn bị ảnh…', 'upload');

  try {
    const files = await collectFiles();
    if (files.length === 0) {
      throw new Error('Chưa có ảnh đầu vào.');
    }

    setProgressUi(0.04, 'Đang tải ảnh lên backend…', 'upload');

    const result = await client.reconstruct(
      files,
      {
        foregroundRatio: Number(fgRatioInput.value),
        mcResolution: Number(qualitySelect.value),
        targetFaces: Number(targetFacesSelect.value),
        removeBackground: removeBgInput.checked,
      },
      {
        onUploadProgress: (ratio) => {
          setProgressUi(0.04 + ratio * 0.14, 'Đang tải ảnh lên backend…', 'upload');
        },
        onStage: (stage, progress, message) => {
          setProgressUi(progress, currentStageLabel(stage, message), stage);
        },
      },
    );

    currentResult = result;
    setProgressUi(0.96, 'Đang tải mô hình vào viewer…', 'loading');
    resultCard.hidden = false;
    resultTime.textContent = `${result.elapsed_seconds.toFixed(1)}s`;
    statVertices.textContent = formatNumber(result.vertices);
    statFaces.textContent = formatNumber(result.faces);

    const loaded = await showModel(result.glb_url, result);
    if (loaded) {
      setProgressUi(1, 'Hoàn tất', 'done');
      progressEl.hidden = true;
      showToast('Mô hình 3D đã sẵn sàng.', 'success');
    } else {
      progressEl.hidden = true;
      showToast('GLB đã tạo — bạn vẫn có thể tải xuống bên dưới.', 'info');
    }
  } catch (error) {
    progressEl.hidden = true;
    showError(errorMessage(error));
    showToast(errorMessage(error), 'error');
  } finally {
    setBusy(false);
  }
}

// --------------------------------------------------------------------- events
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

for (const eventName of ['dragenter', 'dragover'] as const) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragging');
  });
}

dropzone.addEventListener('dragleave', (event) => {
  const related = event.relatedTarget;
  if (related instanceof Node && dropzone.contains(related)) return;
  dropzone.classList.remove('dragging');
});

dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
  if (event.dataTransfer?.files) {
    void addFiles(event.dataTransfer.files);
  }
});

window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

fileInput.addEventListener('change', () => {
  if (fileInput.files) {
    void addFiles(fileInput.files);
  }
  fileInput.value = '';
});

sliceToggle.addEventListener('change', () => {
  if (sliceToggle.checked) {
    enableSlicer();
  } else {
    disableSlicer();
  }
});

gridRowsInput.addEventListener('change', applyGridSize);
gridColsInput.addEventListener('change', applyGridSize);

sliceCanvas.addEventListener('click', (event) => {
  if (!sliceBitmap || !sliceEnabled) return;
  const bounds = sliceCanvas.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return;
  const x = (event.clientX - bounds.left) / bounds.width;
  const y = (event.clientY - bounds.top) / bounds.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  const column = Math.min(gridColumns - 1, Math.floor(x * gridColumns));
  const row = Math.min(gridRows - 1, Math.floor(y * gridRows));
  const index = gridIndex(row, column, gridColumns);
  if (selectedCells.has(index)) {
    selectedCells.delete(index);
  } else {
    selectedCells.add(index);
  }
  drawSlicer();
  updateGenerateState();
});

fgRatioInput.addEventListener('input', () => {
  fgRatioValue.textContent = Number(fgRatioInput.value).toFixed(2);
});

generateBtn.addEventListener('click', () => {
  void generate();
});

downloadBtn.addEventListener('click', () => {
  if (!currentResult || !downloadClient) return;
  downloadBtn.disabled = true;
  void downloadClient
    .download(currentResult.download_url, currentResult.file_name)
    .then(() => showToast('Đang tải file GLB…', 'success'))
    .catch((error: unknown) => showToast(errorMessage(error), 'error'))
    .finally(() => {
      downloadBtn.disabled = false;
    });
});

displayModes.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const button = target.closest<HTMLButtonElement>('button[data-mode]');
  if (!button) return;
  const mode = button.dataset.mode;
  if (mode !== 'pbr' && mode !== 'wireframe' && mode !== 'clay') return;
  studio.setDisplayMode(mode as DisplayMode);
  for (const item of displayModes.querySelectorAll<HTMLButtonElement>('button')) {
    item.classList.toggle('active', item === button);
  }
});

envIntensity.addEventListener('input', () => {
  const value = Number(envIntensity.value);
  studio.setEnvironmentIntensity(value);
  envIntensity.title = `Environment intensity: ${value.toFixed(2)}`;
});

autoRotateBtn.addEventListener('click', () => {
  const next = autoRotateBtn.getAttribute('aria-pressed') !== 'true';
  autoRotateBtn.setAttribute('aria-pressed', String(next));
  autoRotateBtn.classList.toggle('active', next);
  studio.setAutoRotate(next);
});

resetBtn.addEventListener('click', () => {
  studio.resetView();
});

apiCheckBtn.addEventListener('click', () => {
  storeApiUrl(apiUrlInput.value.trim());
  void checkHealth(true);
});

apiUrlInput.addEventListener('input', () => {
  storeApiUrl(apiUrlInput.value.trim());
});

apiUrlInput.addEventListener('change', () => {
  void checkHealth(false);
});

apiUrlInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    void checkHealth(true);
  }
});

demoBtn.addEventListener('click', () => {
  void loadDemo();
});

window.addEventListener('resize', () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (sliceEnabled) drawSlicer();
  }, 150);
});

window.addEventListener('beforeunload', () => {
  studio.dispose();
  for (const item of uploads) {
    URL.revokeObjectURL(item.url);
  }
  sliceBitmap?.close();
});

// ----------------------------------------------------------------- init
apiUrlInput.value = readStoredApiUrl();
fgRatioValue.textContent = Number(fgRatioInput.value).toFixed(2);
updateSliceCount();
renderUploads();
setApiStatus('idle', 'Chưa kiểm tra kết nối');
void checkHealth(false);

const initialModel = new URLSearchParams(window.location.search).get('model');
if (initialModel) {
  const resolved = new URL(initialModel, window.location.href).toString();
  stageHint.textContent = 'Mô hình từ ?model= · kéo để xoay';
  void showModel(resolved, null);
}
