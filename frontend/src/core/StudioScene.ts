import {
  ACESFilmicToneMapping,
  Box3,
  DirectionalLight,
  Group,
  MathUtils,
  Mesh,
  MeshBasicMaterial,
  MeshMatcapMaterial,
  MeshStandardMaterial,
  Object3D,
  PCFSoftShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  PMREMGenerator,
  Scene,
  ShadowMaterial,
  SRGBColorSpace,
  Texture,
  Vector3,
  WebGLRenderer,
  type Material,
  type WebGLRenderTarget,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';

import type { DisplayMode, ModelStats } from '../types';
import { createClayMatcap, createGroundGlowTexture } from './textures';

const MODEL_MAX_SIZE = 2;
const HOME_DIRECTION = new Vector3(1, 0.44, 1).normalize();
const RESET_DURATION_MS = 450;

interface CameraTween {
  startedAt: number;
  fromPosition: Vector3;
  toPosition: Vector3;
  fromTarget: Vector3;
  toTarget: Vector3;
}

function isMesh(object: Object3D): object is Mesh {
  return (object as Mesh).isMesh === true;
}

function disposeMaterial(material: Material): void {
  const record = material as unknown as Record<string, unknown>;
  for (const value of Object.values(record)) {
    if (value instanceof Texture) {
      value.dispose();
    }
  }
  material.dispose();
}

function disposeObject3D(root: Object3D): void {
  root.traverse((object) => {
    if (!isMesh(object)) return;
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      disposeMaterial(material);
    }
  });
}

export function computeModelStats(root: Object3D): ModelStats {
  let vertices = 0;
  let triangles = 0;
  root.traverse((object) => {
    if (!isMesh(object)) return;
    const position = object.geometry.getAttribute('position');
    if (!position) return;
    vertices += position.count;
    const index = object.geometry.getIndex();
    triangles += Math.floor((index ? index.count : position.count) / 3);
  });
  return { vertices, triangles };
}

/**
 * Studio-quality Three.js viewer: ACES tone mapping, room IBL, three-point
 * lighting, soft shadows, material modes and automatic camera framing.
 */
export class StudioScene {
  private readonly container: HTMLElement;
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera: PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly loader = new GLTFLoader();

  private readonly keyLight: DirectionalLight;
  private readonly fillLight: DirectionalLight;
  private readonly rimLight: DirectionalLight;
  private readonly ground: Mesh;
  private readonly groundGlow: Mesh;
  private readonly groundGlowTexture: Texture;
  private readonly envRenderTarget: WebGLRenderTarget;
  private readonly wireframeMaterial: MeshBasicMaterial;
  private readonly clayMaterial: MeshMatcapMaterial;

  private readonly originalMaterials = new Map<Mesh, Material | Material[]>();
  private readonly homePosition = new Vector3(3.4, 2.3, 4.4);
  private readonly homeTarget = new Vector3(0, 0, 0);

  private modelRoot: Group | null = null;
  private animationFrame = 0;
  private resizeObserver: ResizeObserver | null = null;
  private cameraTween: CameraTween | null = null;
  private currentMode: DisplayMode = 'pbr';
  private loadToken = 0;
  private disposed = false;

  constructor(container: HTMLElement, canvas: HTMLCanvasElement) {
    this.container = container;

    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);

    this.renderer = new WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;

    this.camera = new PerspectiveCamera(38, width / height, 0.01, 100);
    this.camera.position.copy(this.homePosition);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.rotateSpeed = 0.8;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.8;
    this.controls.screenSpacePanning = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.2;
    this.controls.minPolarAngle = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.52;
    this.controls.target.copy(this.homeTarget);
    this.controls.update();

    // Room IBL makes metals/ceramics read correctly and avoids black shadows.
    const pmrem = new PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    this.envRenderTarget = pmrem.fromScene(room, 0.04);
    this.scene.environment = this.envRenderTarget.texture;
    this.scene.environmentIntensity = 1;
    room.dispose();
    pmrem.dispose();

    this.keyLight = new DirectionalLight(0xfff2e0, 2.5);
    this.keyLight.position.set(4.2, 6.2, 4.0);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0004;
    this.keyLight.shadow.normalBias = 0.025;
    this.keyLight.shadow.camera.near = 0.1;
    this.keyLight.shadow.camera.far = 30;
    this.keyLight.target.position.set(0, 0, 0);

    this.fillLight = new DirectionalLight(0xa8c6ff, 0.95);
    this.fillLight.position.set(-4.5, 2.6, 3.4);

    this.rimLight = new DirectionalLight(0xc8ecff, 2.1);
    this.rimLight.position.set(-1.8, 3.4, -5.0);

    this.scene.add(this.keyLight, this.keyLight.target, this.fillLight, this.rimLight);

    this.ground = new Mesh(
      new PlaneGeometry(24, 24),
      new ShadowMaterial({ color: 0x000000, opacity: 0.46 }),
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.ground.renderOrder = 1;

    this.groundGlowTexture = createGroundGlowTexture();
    this.groundGlow = new Mesh(
      new PlaneGeometry(1, 1),
      new MeshBasicMaterial({
        map: this.groundGlowTexture,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      }),
    );
    this.groundGlow.rotation.x = -Math.PI / 2;
    this.groundGlow.position.y = -0.002;
    this.groundGlow.renderOrder = 0;

    this.scene.add(this.ground, this.groundGlow);

    const matcap = createClayMatcap();
    this.wireframeMaterial = new MeshBasicMaterial({
      color: 0x7cf5d4,
      wireframe: true,
      transparent: true,
      opacity: 0.7,
    });
    this.clayMaterial = new MeshMatcapMaterial({
      matcap,
      color: 0xd9d4c8,
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);

    this.resize();
    this.animationFrame = requestAnimationFrame(this.animate);
  }

  // ------------------------------------------------------------------ loading
  async loadFromUrl(
    url: string,
    onProgress?: (ratio: number) => void,
  ): Promise<ModelStats> {
    const token = ++this.loadToken;

    const gltf = await new Promise<GLTF>((resolve, reject) => {
      this.loader.load(
        url,
        resolve,
        (event: ProgressEvent) => {
          if (onProgress && event.lengthComputable && event.total > 0) {
            onProgress(Math.min(1, event.loaded / event.total));
          }
        },
        (error: unknown) => {
          reject(error instanceof Error ? error : new Error('Không tải được GLB.'));
        },
      );
    });

    if (this.disposed) {
      disposeObject3D(gltf.scene);
      throw new Error('Viewer đã đóng.');
    }
    if (token !== this.loadToken) {
      disposeObject3D(gltf.scene);
      throw new Error('Yêu cầu tải mô hình đã bị thay thế.');
    }

    this.clearModel();
    const root = gltf.scene;
    this.prepareRoot(root);
    this.modelRoot = root;
    this.scene.add(root);
    this.frameModel(root);
    return computeModelStats(root);
  }

  private prepareRoot(root: Object3D): void {
    const extras: Object3D[] = [];
    root.traverse((object) => {
      const candidate = object as Object3D & { isLight?: boolean; isCamera?: boolean };
      if (candidate.isLight || candidate.isCamera) {
        extras.push(object);
      }
    });
    for (const extra of extras) {
      extra.removeFromParent();
    }

    const maxAnisotropy = Math.min(
      8,
      this.renderer.capabilities.getMaxAnisotropy(),
    );

    root.traverse((object) => {
      if (!isMesh(object)) return;
      const mesh = object;
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      if (!mesh.geometry.getAttribute('normal')) {
        mesh.geometry.computeVertexNormals();
      }

      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        if (material instanceof MeshStandardMaterial) {
          material.envMapIntensity = 1;
          material.needsUpdate = true;
        }
        const record = material as unknown as Record<string, unknown>;
        for (const value of Object.values(record)) {
          if (value instanceof Texture) {
            value.anisotropy = maxAnisotropy;
          }
        }
      }

      this.originalMaterials.set(mesh, mesh.material);
    });

    this.applyMaterialMode();
  }

  private clearModel(): void {
    if (!this.modelRoot) return;
    for (const [mesh, original] of this.originalMaterials) {
      mesh.material = original;
    }
    this.originalMaterials.clear();
    this.scene.remove(this.modelRoot);
    disposeObject3D(this.modelRoot);
    this.modelRoot = null;
  }

  // ------------------------------------------------------------------ framing
  private frameModel(root: Object3D): void {
    root.updateMatrixWorld(true);
    const box = new Box3().setFromObject(root);
    if (box.isEmpty()) {
      throw new Error('Mô hình không có hình học để hiển thị.');
    }

    const size = box.getSize(new Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(maxDim) || maxDim <= 0) {
      throw new Error('Kích thước mô hình không hợp lệ.');
    }

    const scale = MODEL_MAX_SIZE / maxDim;
    const center = box.getCenter(new Vector3());
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
    root.updateMatrixWorld(true);

    const fitted = new Box3().setFromObject(root);
    const fittedSize = fitted.getSize(new Vector3());
    const radius = Math.max(
      0.4,
      0.5 * Math.max(fittedSize.x, fittedSize.y, fittedSize.z),
    );

    const floorY = fitted.min.y - 0.003;
    this.ground.position.y = floorY;
    this.groundGlow.position.y = floorY - 0.005;
    this.groundGlow.scale.setScalar(Math.max(3.4, radius * 7));

    const verticalFov = MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
    const fitFov = Math.max(0.15, Math.min(verticalFov, horizontalFov));
    const distance = Math.max(2.2, (radius / Math.sin(fitFov / 2)) * 1.35);

    this.homePosition.copy(HOME_DIRECTION).multiplyScalar(distance);
    this.homeTarget.set(0, 0, 0);

    this.camera.near = Math.max(0.01, distance / 200);
    this.camera.far = Math.max(60, distance * 12);
    this.camera.position.copy(this.homePosition);
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(this.homeTarget);

    this.controls.target.copy(this.homeTarget);
    this.controls.minDistance = Math.max(0.6, distance * 0.4);
    this.controls.maxDistance = Math.max(4, distance * 3.5);
    this.controls.update();

    const shadowCamera = this.keyLight.shadow.camera;
    const extent = Math.max(1.6, radius * 2.5);
    shadowCamera.left = -extent;
    shadowCamera.right = extent;
    shadowCamera.top = extent * 1.6;
    shadowCamera.bottom = -extent;
    shadowCamera.near = 0.1;
    shadowCamera.far = distance * 6;
    shadowCamera.updateProjectionMatrix();
  }

  // -------------------------------------------------------------------- modes
  setDisplayMode(mode: DisplayMode): void {
    this.currentMode = mode;
    this.applyMaterialMode();
  }

  private applyMaterialMode(): void {
    for (const [mesh, original] of this.originalMaterials) {
      if (this.currentMode === 'wireframe') {
        mesh.material = this.wireframeMaterial;
      } else if (this.currentMode === 'clay') {
        mesh.material = this.clayMaterial;
      } else {
        mesh.material = original;
      }
    }
  }

  setEnvironmentIntensity(value: number): void {
    this.scene.environmentIntensity = MathUtils.clamp(value, 0, 3);
  }

  setAutoRotate(enabled: boolean): void {
    this.controls.autoRotate = enabled;
  }

  resetView(): void {
    this.cameraTween = {
      startedAt: performance.now(),
      fromPosition: this.camera.position.clone(),
      toPosition: this.homePosition.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: this.homeTarget.clone(),
    };
    this.controls.enabled = false;
  }

  getStats(): ModelStats {
    return this.modelRoot
      ? computeModelStats(this.modelRoot)
      : { vertices: 0, triangles: 0 };
  }

  // ------------------------------------------------------------------- render
  private readonly animate = (): void => {
    if (this.disposed) return;
    this.animationFrame = requestAnimationFrame(this.animate);

    if (this.cameraTween) {
      const elapsed = performance.now() - this.cameraTween.startedAt;
      const t = Math.min(1, elapsed / RESET_DURATION_MS);
      const eased = 1 - Math.pow(1 - t, 3);
      this.camera.position.lerpVectors(
        this.cameraTween.fromPosition,
        this.cameraTween.toPosition,
        eased,
      );
      this.controls.target.lerpVectors(
        this.cameraTween.fromTarget,
        this.cameraTween.toTarget,
        eased,
      );
      this.camera.lookAt(this.controls.target);
      if (t >= 1) {
        this.cameraTween = null;
        this.controls.enabled = true;
        this.controls.update();
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  private resize(): void {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.controls.dispose();
    this.clearModel();

    this.scene.remove(this.ground, this.groundGlow);
    this.ground.geometry.dispose();
    disposeMaterial(this.ground.material as Material);
    this.groundGlow.geometry.dispose();
    disposeMaterial(this.groundGlow.material as Material);

    this.wireframeMaterial.dispose();
    disposeMaterial(this.clayMaterial);

    this.scene.environment = null;
    this.envRenderTarget.dispose();
    this.scene.clear();
    this.renderer.dispose();
  }
}
