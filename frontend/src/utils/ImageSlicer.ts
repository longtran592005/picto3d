import type { SlicedView } from '../types';

export interface GridSpec {
  rows: number;
  columns: number;
}

export interface GridSelection extends GridSpec {
  selected: Set<number>;
}

const MAX_SLICE_EDGE = 1024;

export function clampGridValue(value: number, fallback = 2): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(1, Math.min(6, Math.round(value)));
}

export function gridIndex(row: number, column: number, columns: number): number {
  return row * columns + column;
}

export function allCells(rows: number, columns: number): Set<number> {
  const cells = new Set<number>();
  for (let index = 0; index < rows * columns; index += 1) {
    cells.add(index);
  }
  return cells;
}

export async function loadImageBitmap(file: File): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file);
  } catch {
    throw new Error('Không đọc được ảnh. Hãy thử PNG, JPG hoặc WEBP.');
  }
}

export function readImageSize(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => {
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      reject(new Error('Không đọc được kích thước ảnh.'));
    };
    image.src = url;
  });
}

export function drawGridPreview(
  canvas: HTMLCanvasElement,
  bitmap: ImageBitmap,
  selection: GridSelection,
): void {
  const bounds = canvas.getBoundingClientRect();
  const displayWidth = Math.max(1, Math.round(bounds.width));
  const displayHeight = Math.max(
    1,
    Math.round((displayWidth * bitmap.height) / Math.max(1, bitmap.width)),
  );
  const dpr = Math.min(2, window.devicePixelRatio || 1);

  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.round(displayWidth * dpr);
  canvas.height = Math.round(displayHeight * dpr);

  const context = canvas.getContext('2d');
  if (!context) return;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, displayWidth, displayHeight);
  context.drawImage(bitmap, 0, 0, displayWidth, displayHeight);

  const rows = Math.max(1, selection.rows);
  const columns = Math.max(1, selection.columns);
  const cellWidth = displayWidth / columns;
  const cellHeight = displayHeight / rows;

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const index = gridIndex(row, column, columns);
      const x = column * cellWidth;
      const y = row * cellHeight;
      const chosen = selection.selected.has(index);

      if (!chosen) {
        context.fillStyle = 'rgba(4, 6, 9, 0.66)';
        context.fillRect(x, y, cellWidth, cellHeight);
      }

      context.strokeStyle = chosen
        ? 'rgba(124, 245, 212, 0.55)'
        : 'rgba(255, 255, 255, 0.16)';
      context.lineWidth = chosen ? 1.4 : 1;
      context.strokeRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);

      const radius = Math.min(11, Math.max(8, Math.min(cellWidth, cellHeight) * 0.16));
      const centerX = x + cellWidth / 2;
      const centerY = y + cellHeight / 2;
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, Math.PI * 2);
      context.fillStyle = chosen ? 'rgba(124, 245, 212, 0.92)' : 'rgba(12, 15, 20, 0.82)';
      context.fill();
      context.strokeStyle = chosen
        ? 'rgba(7, 17, 14, 0.35)'
        : 'rgba(255, 255, 255, 0.2)';
      context.lineWidth = 1;
      context.stroke();

      context.fillStyle = chosen ? '#07110e' : 'rgba(210, 220, 232, 0.8)';
      context.font = '600 10px Inter, Segoe UI, sans-serif';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(String(index + 1), centerX, centerY + 0.5);
    }
  }
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/png');
  });
}

export async function sliceImage(
  bitmap: ImageBitmap,
  selection: GridSelection,
  namePrefix = 'view',
): Promise<SlicedView[]> {
  const rows = Math.max(1, selection.rows);
  const columns = Math.max(1, selection.columns);
  const indices = [...selection.selected].sort((a, b) => a - b);
  if (indices.length === 0) return [];

  const cellWidth = bitmap.width / columns;
  const cellHeight = bitmap.height / rows;
  const views: SlicedView[] = [];

  for (const index of indices) {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const sx = Math.round(column * cellWidth);
    const sy = Math.round(row * cellHeight);
    const sw = Math.max(1, Math.round((column + 1) * cellWidth) - sx);
    const sh = Math.max(1, Math.round((row + 1) * cellHeight) - sy);

    const scale = Math.min(1, MAX_SLICE_EDGE / Math.max(sw, sh));
    const outWidth = Math.max(1, Math.round(sw * scale));
    const outHeight = Math.max(1, Math.round(sh * scale));

    const canvas = document.createElement('canvas');
    canvas.width = outWidth;
    canvas.height = outHeight;
    const context = canvas.getContext('2d');
    if (!context) continue;
    context.imageSmoothingQuality = 'high';
    context.drawImage(bitmap, sx, sy, sw, sh, 0, 0, outWidth, outHeight);

    const blob = await canvasToBlob(canvas);
    if (!blob) continue;

    views.push({
      file: new File([blob], `${namePrefix}_r${row + 1}c${column + 1}.png`, {
        type: 'image/png',
      }),
      row,
      column,
      index,
    });
  }

  return views;
}
