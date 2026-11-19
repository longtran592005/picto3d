import {
  CanvasTexture,
  LinearFilter,
  SRGBColorSpace,
  type Texture,
} from 'three';

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas 2D không khả dụng trên trình duyệt này.');
  }
  return context;
}

/**
 * Procedural clay matcap: warm top-left key, cool rim, soft lower bounce.
 * Avoids shipping an external texture while still reading like a studio matcap.
 */
export function createClayMatcap(size = 256): Texture {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = context2d(canvas);

  context.fillStyle = '#15181d';
  context.fillRect(0, 0, size, size);

  const base = context.createRadialGradient(
    size * 0.42,
    size * 0.36,
    size * 0.02,
    size * 0.5,
    size * 0.52,
    size * 0.55,
  );
  base.addColorStop(0, '#fffdf6');
  base.addColorStop(0.28, '#eee7d8');
  base.addColorStop(0.62, '#b9b1a2');
  base.addColorStop(0.86, '#6f6a61');
  base.addColorStop(1, '#2b2c2c');
  context.fillStyle = base;
  context.fillRect(0, 0, size, size);

  const rim = context.createRadialGradient(
    size * 0.5,
    size * 0.5,
    size * 0.38,
    size * 0.5,
    size * 0.5,
    size * 0.52,
  );
  rim.addColorStop(0, 'rgba(143, 184, 255, 0)');
  rim.addColorStop(0.72, 'rgba(143, 184, 255, 0.10)');
  rim.addColorStop(1, 'rgba(176, 214, 255, 0.42)');
  context.fillStyle = rim;
  context.fillRect(0, 0, size, size);

  const shade = context.createRadialGradient(
    size * 0.78,
    size * 0.8,
    size * 0.02,
    size * 0.78,
    size * 0.8,
    size * 0.42,
  );
  shade.addColorStop(0, 'rgba(8, 10, 12, 0.55)');
  shade.addColorStop(1, 'rgba(8, 10, 12, 0)');
  context.fillStyle = shade;
  context.fillRect(0, 0, size, size);

  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/** Soft pool of light on the floor so the object feels grounded. */
export function createGroundGlowTexture(size = 512): Texture {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = context2d(canvas);
  context.clearRect(0, 0, size, size);

  const gradient = context.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2,
  );
  gradient.addColorStop(0, 'rgba(126, 156, 186, 0.30)');
  gradient.addColorStop(0.32, 'rgba(104, 132, 162, 0.16)');
  gradient.addColorStop(0.68, 'rgba(74, 96, 120, 0.05)');
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);

  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.needsUpdate = true;
  return texture;
}
