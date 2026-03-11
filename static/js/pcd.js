import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";

const loader = new PCDLoader();
const pointsCache = new Map();

const CAMERA_PRESETS = {
  default: {
    distScale: 0.54,
    distMin: 0.67,
    leftScale: 0.28,
    leftMin: 0.22,
    downScale: 0.18,
    downMin: 0.24,
    targetDownScale: 0.0,
    targetDownMin: 0.0,
    targetShiftLeftScale: 0.0,
    targetShiftLeftMin: 0.0,
    yawLeftDeg: 0,
    zoomMul: 1.0,
    fovDeg: 60,
    cameraZoom: 1.0,
    pointYawLeftDeg: 0,
    pointShiftRight: 0.0,
    pointShiftToward: 0.0,
  },
  "14build_7floor": {
    distScale: 0.50,
    distMin: 0.62,
    leftScale: 0.24,
    leftMin: 0.20,
    downScale: 0.14,
    downMin: 0.18,
    targetDownScale: 0.16,
    targetDownMin: 0.20,
    targetShiftLeftScale: 0.0,
    targetShiftLeftMin: 0.0,
    yawLeftDeg: 0,
    zoomMul: 1.1,
    fovDeg: 58,
    cameraZoom: 1.05,
    pointYawLeftDeg: 0,
    pointShiftRight: 0.0,
    pointShiftToward: 0.0,
  },
  "34build": {
    distScale: 0.50,
    distMin: 0.62,
    leftScale: 0.24,
    leftMin: 0.20,
    downScale: 0.14,
    downMin: 0.18,
    targetDownScale: 0.16,
    targetDownMin: 0.20,
    targetShiftLeftScale: 0.0,
    targetShiftLeftMin: 0.0,
    yawLeftDeg: -32,
    zoomMul: 1.1,
    fovDeg: 58,
    cameraZoom: 1.05,
    pointYawLeftDeg: 0,
    pointShiftRight: 0.9,
    pointShiftToward: 0.0,
  },
  "22build": {
    distScale: 0.34,
    distMin: 0.38,
    leftScale: 1.04,
    leftMin: 0.98,
    downScale: 0.06,
    downMin: 0.10,
    targetDownScale: 0.12,
    targetDownMin: 0.16,
    targetShiftLeftScale: 0.0,
    targetShiftLeftMin: 0.0,
    yawLeftDeg: 0,
    zoomMul: 5.6,
    fovDeg: 28,
    cameraZoom: 1.85,
    pointYawLeftDeg: 48,
    pointShiftRight: 0.8,
    pointShiftToward: 6.1,
  },
  "36build": {
    distScale: 0.34,
    distMin: 0.40,
    leftScale: 0.68,
    leftMin: 0.50,
    downScale: 0.04,
    downMin: 0.06,
    targetDownScale: 0.14,
    targetDownMin: 0.18,
    targetShiftLeftScale: 0.16,
    targetShiftLeftMin: 0.22,
    yawLeftDeg: 40,
    zoomMul: 4.8,
    fovDeg: 42,
    cameraZoom: 2.4,
    pointYawLeftDeg: 15,
    pointShiftRight: 0.0,
    pointShiftToward: 0.0,
  },
};

const DISPLAY_AXIS_CORRECTION = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
);

const RESULT_METHOD_ORDER = ["GT", "marmot", "lindell", "ours"];
const RESULT_SCENES = {
  "14build_7floor": {
    GT: "GT_14build_7floor_hist001_20250912173505_t01757666105461000000_000000_gt.pcd",
    marmot: "marmot_14build_7floor_hist001_20250912173505_t01757666105461000000_000000.pcd",
    lindell: "lindell_14build_7floor_hist001_20250912173505_t01757666105461000000_000000.pcd",
    ours: "ours_14build_7floor_hist001_20250912173505_t01757666105461000000_000000.pcd",
  },
  "22build": {
    GT: "GT_22build_hist001_20250901152822_t01756708103212000000_000000_gt.pcd",
    marmot: "marmot_22build_hist001_20250901152822_t01756708103212000000_000000.pcd",
    lindell: "lindell_22build_hist001_20250901152822_t01756708103212000000_000000.pcd",
    ours: "ours_22build_hist001_20250901152822_t01756708103212000000_000000.pcd",
  },
  "36build": {
    GT: "GT_36build_hist002_20250909150845_t01757398125644000000_000000_gt.pcd",
    marmot: "marmot_36build_hist002_20250909150845_t01757398125644000000_000000.pcd",
    lindell: "lindell_36build_hist002_20250909150845_t01757398125644000000_000000.pcd",
    ours: "ours_36build_hist002_20250909150845_t01757398125644000000_000000.pcd",
  },
};
const RESULT_SCENE_DISPLAY_LABELS = {
  "14build_7floor": "Scene002",
  "22build": "Scene007",
  "36build": "Scene009",
};

const DATASET_METHOD_ORDER = ["ghost-fwl"];
const RESULT_LIKE_DATASET_SCENES = new Set(["14build_7floor", "22build", "36build"]);
const DATASET_SCENES = {
  "14build_2floor": { "ghost-fwl": "ghost-fwl/14build_2floor.pcd" },
  "14build_7floor": { "ghost-fwl": "ghost-fwl/14build_7floor.pcd" },
  "16buildA_mid": { "ghost-fwl": "ghost-fwl/16buildA_mid.pcd" },
  "16buildA_large": { "ghost-fwl": "ghost-fwl/16buildA_large.pcd" },
  "11build": { "ghost-fwl": "ghost-fwl/11build.pcd" },
  "16build": { "ghost-fwl": "ghost-fwl/16build.pcd" },
  "22build": { "ghost-fwl": "ghost-fwl/22build.pcd" },
  "34build": { "ghost-fwl": "ghost-fwl/34build.pcd" },
  "36build": { "ghost-fwl": "ghost-fwl/36build.pcd" },
  "gym_build": { "ghost-fwl": "ghost-fwl/gym_build.pcd" },
};
const DATASET_SCENE_DISPLAY_LABELS = {
  "14build_2floor": "Scene001",
  "14build_7floor": "Scene002",
  "16buildA_mid": "Scene003",
  "16buildA_large": "Scene004",
  "11build": "Scene005",
  "16build": "Scene006",
  "22build": "Scene007",
  "34build": "Scene008",
  "36build": "Scene009",
  "gym_build": "Scene010",
};
const DATASET_MIN_VISIBLE_DISTANCE_M = 0.8;
const DATASET_DISABLE_PER_SCENE_YAW = true;
const DATASET_YAW_OVERRIDES = {
  "34build": -10,
  "16buildA_large": -12,
  "16buildA_mid": -6,
};
const DATASET_DOWN_OVERRIDES = {
  "16buildA_mid": {
    downScale: 0.08,
    downMin: 0.08,
  },
  "36build": {
    distScale: 0.12,
    distMin: 0.20,
    downScale: 0.04,
    downMin: 0.06,
  },
  "22build": {
    zoomMul: -0.35,
    cameraZoom: -0.15,
  },
};
const DATASET_PAN_OVERRIDES = {
  "11build": {
    rightScale: -0.08,
    rightMin: -0.08,
    upScale: 0.05,
    upMin: 0.05,
  },
  "34build": {
    rightScale: 0.32,
    rightMin: 0.32,
    upScale: 0.11,
    upMin: 0.11,
  },
  "16buildA_large": {
    rightScale: -0.08,
    rightMin: -0.08,
    upScale: 0.06,
    upMin: 0.06,
  },
  "16buildA_mid": {
    rightScale: 0.18,
    rightMin: 0.18,
    upScale: 0.06,
    upMin: 0.06,
  },
  "22build": {
    rightScale: 0.07,
    rightMin: 0.07,
    upScale: 0.05,
    upMin: 0.05,
  },
  "gym_build": {
    rightScale: -0.04,
    rightMin: -0.04,
    upScale: 0.08,
    upMin: 0.08,
  },
};
const DATASET_CAMERA_PRESET_SCENE_KEY = "14build_7floor";
const DATASET_CAMERA_YAW_RIGHT_DEG = 30;
const DATASET_CAMERA_UPSIDE_DOWN = true;
const DATASET_CAMERA_DOWN_SCALE_BONUS = 0.34;
const DATASET_CAMERA_DOWN_MIN_BONUS = 0.34;
const DATASET_CAMERA_TARGET_DOWN_SCALE_BONUS = 0.0;
const DATASET_CAMERA_TARGET_DOWN_MIN_BONUS = 0.0;
const DATASET_CAMERA_PAN_UP_SCALE = 0.24;
const DATASET_CAMERA_PAN_UP_MIN = 0.24;
const DATASET_CAMERA_PAN_RIGHT_SCALE = 0.10;
const DATASET_CAMERA_PAN_RIGHT_MIN = 0.10;

function getCameraPreset(sceneKey) {
  const sceneOverride = DATASET_DOWN_OVERRIDES[sceneKey] ?? {};
  const usesDatasetOverrides = DATASET_SCENES[sceneKey] && !RESULT_LIKE_DATASET_SCENES.has(sceneKey);
  if (usesDatasetOverrides) {
    const basePreset = CAMERA_PRESETS[sceneKey]
      ?? CAMERA_PRESETS[DATASET_CAMERA_PRESET_SCENE_KEY]
      ?? CAMERA_PRESETS.default;
    const sceneYawLeftDeg = DATASET_DISABLE_PER_SCENE_YAW ? 0 : (basePreset.yawLeftDeg ?? 0);
    const datasetYawOverride = DATASET_YAW_OVERRIDES[sceneKey] ?? 0;
    return {
      ...basePreset,
      yawLeftDeg: sceneYawLeftDeg - DATASET_CAMERA_YAW_RIGHT_DEG + datasetYawOverride,
      distScale: (basePreset.distScale ?? 0) + (sceneOverride.distScale ?? 0),
      distMin: (basePreset.distMin ?? 0) + (sceneOverride.distMin ?? 0),
      zoomMul: (basePreset.zoomMul ?? 1.0) + (sceneOverride.zoomMul ?? 0),
      cameraZoom: (basePreset.cameraZoom ?? 1.0) + (sceneOverride.cameraZoom ?? 0),
      downScale: (basePreset.downScale ?? 0) + DATASET_CAMERA_DOWN_SCALE_BONUS + (sceneOverride.downScale ?? 0),
      downMin: (basePreset.downMin ?? 0) + DATASET_CAMERA_DOWN_MIN_BONUS + (sceneOverride.downMin ?? 0),
      targetDownScale: (basePreset.targetDownScale ?? 0) + DATASET_CAMERA_TARGET_DOWN_SCALE_BONUS,
      targetDownMin: (basePreset.targetDownMin ?? 0) + DATASET_CAMERA_TARGET_DOWN_MIN_BONUS,
    };
  }
  const basePreset = CAMERA_PRESETS[sceneKey] ?? CAMERA_PRESETS.default;
  return {
    ...basePreset,
    distScale: (basePreset.distScale ?? 0) + (sceneOverride.distScale ?? 0),
    distMin: (basePreset.distMin ?? 0) + (sceneOverride.distMin ?? 0),
    zoomMul: (basePreset.zoomMul ?? 1.0) + (sceneOverride.zoomMul ?? 0),
    cameraZoom: (basePreset.cameraZoom ?? 1.0) + (sceneOverride.cameraZoom ?? 0),
    downScale: (basePreset.downScale ?? 0) + (sceneOverride.downScale ?? 0),
    downMin: (basePreset.downMin ?? 0) + (sceneOverride.downMin ?? 0),
  };
}

function fitCameraToBox(camera, controls, box, sceneKey) {
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());

  camera.near = Math.max(size / 1000, 0.001);
  camera.far = size * 10 + 10;
  camera.updateProjectionMatrix();

  const preset = getCameraPreset(sceneKey);
  const zoomMul = Math.max(preset.zoomMul ?? 1.0, 0.1);
  const dist = Math.max(size * preset.distScale, preset.distMin) / zoomMul;
  const leftOffset = Math.max(size * preset.leftScale, preset.leftMin);
  const downOffset = Math.max(size * preset.downScale, preset.downMin);
  const targetDownOffset = Math.max(size * preset.targetDownScale, preset.targetDownMin);
  const targetShiftLeftOffset = Math.max(size * preset.targetShiftLeftScale, preset.targetShiftLeftMin);
  const yawLeftRad = THREE.MathUtils.degToRad(preset.yawLeftDeg ?? 0);
  camera.fov = preset.fovDeg ?? 60;
  camera.zoom = preset.cameraZoom ?? 1.0;
  camera.updateProjectionMatrix();

  const isDatasetScene = Boolean(DATASET_SCENES[sceneKey] && !RESULT_LIKE_DATASET_SCENES.has(sceneKey));
  const datasetPanUpOffset = isDatasetScene
    ? Math.max(size * DATASET_CAMERA_PAN_UP_SCALE, DATASET_CAMERA_PAN_UP_MIN)
    : 0;
  const datasetPanRightOffset = isDatasetScene
    ? Math.max(size * DATASET_CAMERA_PAN_RIGHT_SCALE, DATASET_CAMERA_PAN_RIGHT_MIN)
    : 0;
  const panOverride = DATASET_PAN_OVERRIDES[sceneKey];
  const datasetPanUpExtra = panOverride
    ? Math.max(size * (panOverride.upScale ?? 0), panOverride.upMin ?? 0)
    : 0;
  const datasetPanRightExtra = panOverride
    ? Math.min(size * (panOverride.rightScale ?? 0), panOverride.rightMin ?? 0)
    : 0;
  const totalDatasetPanUpOffset = datasetPanUpOffset + datasetPanUpExtra;
  const totalDatasetPanRightOffset = datasetPanRightOffset + datasetPanRightExtra;
  controls.target.copy(center).add(
    new THREE.Vector3(0, -targetShiftLeftOffset + totalDatasetPanRightOffset, targetDownOffset - totalDatasetPanUpOffset),
  );
  const upZ = isDatasetScene && DATASET_CAMERA_UPSIDE_DOWN ? 1 : -1;
  camera.up.set(0, 0, upZ);
  const relativePosition = new THREE.Vector3(-dist, -leftOffset, downOffset);
  if (yawLeftRad !== 0) {
    relativePosition.applyAxisAngle(camera.up, yawLeftRad);
  }
  camera.position.copy(center).add(relativePosition).add(
    new THREE.Vector3(0, totalDatasetPanRightOffset, -totalDatasetPanUpOffset),
  );
  controls.update();
}

function fitCameraToObjects(camera, controls, objects, sceneKey) {
  const box = new THREE.Box3();
  let hasObject = false;
  for (const object of objects) {
    if (!object) continue;
    box.union(new THREE.Box3().setFromObject(object));
    hasObject = true;
  }
  if (!hasObject || box.isEmpty()) return;
  fitCameraToBox(camera, controls, box, sceneKey);
}

function applyPointSize(points) {
  points.material.size = 0.02;
  points.material.sizeAttenuation = true;
  const hasVertexColor = Boolean(points.geometry?.getAttribute?.("color"));
  points.material.vertexColors = hasVertexColor;
  if (!hasVertexColor) {
    points.material.color.set(0x111111);
  }
  points.material.needsUpdate = true;
}

function isRedPoint(r, g, b) {
  const scale = r > 1 || g > 1 || b > 1 ? 255 : 1;
  const rn = r / scale;
  const gn = g / scale;
  const bn = b / scale;
  const redDominant = rn > gn * 1.35 && rn > bn * 1.35;
  return redDominant && rn >= 0.35;
}

function buildFilteredGeometry(geometry, predicate) {
  const positionAttr = geometry?.getAttribute?.("position");
  const count = positionAttr?.count ?? 0;
  if (!positionAttr || count === 0) return null;

  const kept = [];
  for (let i = 0; i < count; i += 1) {
    if (predicate(i)) kept.push(i);
  }
  if (kept.length === count || kept.length === 0) return null;

  const filtered = new THREE.BufferGeometry();
  for (const [name, attr] of Object.entries(geometry.attributes)) {
    const itemSize = attr.itemSize;
    const TypedArrayCtor = attr.array.constructor;
    const out = new TypedArrayCtor(kept.length * itemSize);
    for (let i = 0; i < kept.length; i += 1) {
      const srcIdx = kept[i] * itemSize;
      const dstIdx = i * itemSize;
      for (let k = 0; k < itemSize; k += 1) {
        out[dstIdx + k] = attr.array[srcIdx + k];
      }
    }
    filtered.setAttribute(name, new THREE.BufferAttribute(out, itemSize, attr.normalized));
  }
  filtered.computeBoundingBox();
  filtered.computeBoundingSphere();
  return filtered;
}

function buildNonRedGeometry(geometry) {
  const colorAttr = geometry?.getAttribute?.("color");
  if (!colorAttr) return null;
  return buildFilteredGeometry(geometry, (i) => {
    const r = colorAttr.getX(i);
    const g = colorAttr.getY(i);
    const b = colorAttr.getZ(i);
    return !isRedPoint(r, g, b);
  });
}

function buildDistanceFilteredGeometry(geometry, minDistance) {
  const positionAttr = geometry?.getAttribute?.("position");
  if (!positionAttr) return null;
  const minDistanceSq = minDistance * minDistance;
  return buildFilteredGeometry(geometry, (i) => {
    const x = positionAttr.getX(i);
    const y = positionAttr.getY(i);
    const z = positionAttr.getZ(i);
    return x * x + y * y + z * z >= minDistanceSq;
  });
}

function applyRedPointVisibility(points, showRed) {
  const fullGeometry = points.userData?.fullGeometry;
  const nonRedGeometry = points.userData?.nonRedGeometry;
  const distanceFilteredGeometry = points.userData?.distanceFilteredGeometry;
  const nonRedDistanceFilteredGeometry = points.userData?.nonRedDistanceFilteredGeometry;
  if (!fullGeometry) return;

  const useDistanceFilter = Boolean(distanceFilteredGeometry);
  const nextGeometry = showRed
    ? (useDistanceFilter ? distanceFilteredGeometry : fullGeometry)
    : (useDistanceFilter ? (nonRedDistanceFilteredGeometry ?? distanceFilteredGeometry) : (nonRedGeometry ?? fullGeometry));

  if (nextGeometry && points.geometry !== nextGeometry) {
    points.geometry = nextGeometry;
  }
}

async function loadPCDFromURL(url) {
  if (pointsCache.has(url)) {
    return pointsCache.get(url);
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch failed: ${res.status} ${res.statusText}`);
  }
  const buf = await res.arrayBuffer();
  const points = loader.parse(buf);
  if (!points?.geometry) {
    throw new Error("PCD parse failed");
  }
  pointsCache.set(url, points);
  return points;
}

function clonePoints(points, sceneKey) {
  const geometry = points.geometry.clone();
  const material = points.material.clone();
  const cloned = new THREE.Points(geometry, material);
  cloned.position.copy(points.position);
  cloned.rotation.copy(points.rotation);
  cloned.scale.copy(points.scale);
  cloned.applyMatrix4(DISPLAY_AXIS_CORRECTION);
  const preset = getCameraPreset(sceneKey);
  const pointYawLeftRad = THREE.MathUtils.degToRad(preset.pointYawLeftDeg ?? 0);
  if (pointYawLeftRad !== 0) {
    cloned.rotateZ(pointYawLeftRad);
  }
  cloned.userData.basePosition = cloned.position.clone();
  cloned.userData.fullGeometry = geometry;
  cloned.userData.nonRedGeometry = buildNonRedGeometry(geometry);
  if (DATASET_SCENES[sceneKey] && !RESULT_LIKE_DATASET_SCENES.has(sceneKey)) {
    cloned.userData.distanceFilteredGeometry = buildDistanceFilteredGeometry(geometry, DATASET_MIN_VISIBLE_DISTANCE_M) ?? geometry;
    cloned.userData.nonRedDistanceFilteredGeometry =
      buildDistanceFilteredGeometry(cloned.userData.nonRedGeometry ?? geometry, DATASET_MIN_VISIBLE_DISTANCE_M);
  }
  return cloned;
}

function applyPointTranslation(points, sceneKey) {
  if (!points) return;
  const preset = getCameraPreset(sceneKey);
  const base = points.userData?.basePosition ?? new THREE.Vector3(0, 0, 0);
  const pointShiftRight = preset.pointShiftRight ?? 0;
  const pointShiftToward = preset.pointShiftToward ?? 0;
  points.position.copy(base);
  points.position.y += pointShiftRight;
  points.position.x -= pointShiftToward;
}

function disposePoints(points) {
  if (!points) return;
  points.userData?.fullGeometry?.dispose?.();
  if (points.userData?.nonRedGeometry && points.userData.nonRedGeometry !== points.userData.fullGeometry) {
    points.userData.nonRedGeometry.dispose();
  }
  if (points.userData?.distanceFilteredGeometry && points.userData.distanceFilteredGeometry !== points.userData.fullGeometry) {
    points.userData.distanceFilteredGeometry.dispose();
  }
  if (
    points.userData?.nonRedDistanceFilteredGeometry
    && points.userData.nonRedDistanceFilteredGeometry !== points.userData.nonRedGeometry
    && points.userData.nonRedDistanceFilteredGeometry !== points.userData.distanceFilteredGeometry
  ) {
    points.userData.nonRedDistanceFilteredGeometry.dispose();
  }
  points.material?.dispose?.();
}

class SingleViewer {
  constructor(root, titleText) {
    this.points = null;
    this.root = document.createElement("div");
    this.root.className = "pcd-compare-window";

    const title = document.createElement("div");
    title.className = "pcd-compare-title";
    title.textContent = titleText;
    this.root.appendChild(title);

    this.viewport = document.createElement("div");
    this.viewport.className = "pcd-compare-viewport";
    this.root.appendChild(this.viewport);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "pcd-compare-canvas";
    this.viewport.appendChild(this.canvas);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100000);
    this.camera.position.set(2, 2, 2);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.enableRotate = false;
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    root.appendChild(this.root);
    this.resize();
  }

  setPointSize() {
    if (this.points) applyPointSize(this.points);
  }

  setRedPointVisibility(showRed) {
    if (this.points) applyRedPointVisibility(this.points, showRed);
  }

  setPoints(rawPoints, { fitView = false, sceneKey, showRed = true } = {}) {
    if (this.points) {
      this.scene.remove(this.points);
      disposePoints(this.points);
    }

    this.points = clonePoints(rawPoints, sceneKey);
    applyRedPointVisibility(this.points, showRed);
    applyPointTranslation(this.points, sceneKey);
    this.scene.add(this.points);
    this.setPointSize();

    if (fitView) {
      fitCameraToObjects(this.camera, this.controls, [this.points], sceneKey);
    }
  }

  resetView(sceneKey) {
    fitCameraToObjects(this.camera, this.controls, [this.points], sceneKey);
    applyPointTranslation(this.points, sceneKey);
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.controls.update();
    this.renderer.setViewport(0, 0, width, height);
    this.renderer.setScissorTest(false);
    this.renderer.render(this.scene, this.camera);
  }
}

class CompareViewer {
  constructor(root, compareMethod) {
    this.compareMethod = compareMethod;
    this.split = 0.5;
    this.leftPoints = null;
    this.rightPoints = null;

    this.root = document.createElement("div");
    this.root.className = "pcd-compare-window";

    const title = document.createElement("div");
    title.className = "pcd-compare-title";
    title.textContent = `${compareMethod} | ours`;
    this.root.appendChild(title);

    this.viewport = document.createElement("div");
    this.viewport.className = "pcd-compare-viewport";
    this.root.appendChild(this.viewport);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "pcd-compare-canvas";
    this.viewport.appendChild(this.canvas);

    this.divider = document.createElement("div");
    this.divider.className = "pcd-divider";
    this.viewport.appendChild(this.divider);

    this.handle = document.createElement("div");
    this.handle.className = "pcd-divider-handle";
    this.divider.appendChild(this.handle);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.leftScene = new THREE.Scene();
    this.leftScene.background = new THREE.Color(0xffffff);
    this.rightScene = new THREE.Scene();
    this.rightScene.background = new THREE.Color(0xffffff);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100000);
    this.camera.position.set(2, 2, 2);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.enableRotate = false;
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    const beginDrag = (event) => {
      event.preventDefault();
      this.setSplitByClientX(event.clientX);
      this.viewport.setPointerCapture(event.pointerId);
      this.dragging = true;
    };
    const onMove = (event) => {
      if (!this.dragging) return;
      this.setSplitByClientX(event.clientX);
    };
    const endDrag = (event) => {
      if (!this.dragging) return;
      this.setSplitByClientX(event.clientX);
      this.dragging = false;
      this.viewport.releasePointerCapture(event.pointerId);
    };

    this.viewport.addEventListener("pointerdown", beginDrag);
    this.viewport.addEventListener("pointermove", onMove);
    this.viewport.addEventListener("pointerup", endDrag);
    this.viewport.addEventListener("pointercancel", endDrag);
    this.viewport.addEventListener("pointerleave", endDrag);

    root.appendChild(this.root);
    this.resize();
    this.updateDivider();
  }

  setSplitByClientX(clientX) {
    const rect = this.viewport.getBoundingClientRect();
    if (!rect.width) return;
    const ratio = (clientX - rect.left) / rect.width;
    this.split = Math.min(0.95, Math.max(0.05, ratio));
    this.updateDivider();
  }

  updateDivider() {
    this.divider.style.left = `${this.split * 100}%`;
  }

  setPointSize() {
    if (this.leftPoints) applyPointSize(this.leftPoints);
    if (this.rightPoints) applyPointSize(this.rightPoints);
  }

  setRedPointVisibility(showRed) {
    if (this.leftPoints) applyRedPointVisibility(this.leftPoints, showRed);
    if (this.rightPoints) applyRedPointVisibility(this.rightPoints, showRed);
  }

  setComparisonPoints(leftRaw, rightRaw, { fitView = false, sceneKey, showRed = true } = {}) {
    if (this.leftPoints) {
      this.leftScene.remove(this.leftPoints);
      disposePoints(this.leftPoints);
    }
    if (this.rightPoints) {
      this.rightScene.remove(this.rightPoints);
      disposePoints(this.rightPoints);
    }

    this.leftPoints = clonePoints(leftRaw, sceneKey);
    this.rightPoints = clonePoints(rightRaw, sceneKey);
    applyRedPointVisibility(this.leftPoints, showRed);
    applyRedPointVisibility(this.rightPoints, showRed);
    applyPointTranslation(this.leftPoints, sceneKey);
    applyPointTranslation(this.rightPoints, sceneKey);

    this.leftScene.add(this.leftPoints);
    this.rightScene.add(this.rightPoints);
    this.setPointSize();

    if (fitView) {
      fitCameraToObjects(this.camera, this.controls, [this.leftPoints, this.rightPoints], sceneKey);
    }
  }

  resetView(sceneKey) {
    fitCameraToObjects(this.camera, this.controls, [this.leftPoints, this.rightPoints], sceneKey);
    applyPointTranslation(this.leftPoints, sceneKey);
    applyPointTranslation(this.rightPoints, sceneKey);
  }

  copyCameraFrom(sourceViewer) {
    this.camera.position.copy(sourceViewer.camera.position);
    this.camera.quaternion.copy(sourceViewer.camera.quaternion);
    this.camera.up.copy(sourceViewer.camera.up);
    this.camera.near = sourceViewer.camera.near;
    this.camera.far = sourceViewer.camera.far;
    this.camera.fov = sourceViewer.camera.fov;
    this.camera.zoom = sourceViewer.camera.zoom;
    this.camera.aspect = sourceViewer.camera.aspect;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(sourceViewer.controls.target);
    this.controls.update();
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;

    this.controls.update();
    const splitPx = Math.floor(width * this.split);
    this.renderer.setScissorTest(true);
    this.renderer.setViewport(0, 0, width, height);

    if (splitPx > 0) {
      this.renderer.setScissor(0, 0, splitPx, height);
      this.renderer.render(this.leftScene, this.camera);
    }

    const rightWidth = width - splitPx;
    if (rightWidth > 0) {
      this.renderer.setScissor(splitPx, 0, rightWidth, height);
      this.renderer.render(this.rightScene, this.camera);
    }

    this.renderer.setScissorTest(false);
  }
}

class PcdApp {
  constructor({
    scenePickerId,
    showRedInputId,
    resetViewBtnId,
    statusId,
    gridId,
    scenes,
    methodOrder,
    sceneDisplayLabels,
    viewerMode,
  }) {
    this.scenePicker = document.getElementById(scenePickerId);
    this.showRedInput = document.getElementById(showRedInputId);
    this.resetViewBtn = document.getElementById(resetViewBtnId);
    this.statusText = document.getElementById(statusId);
    this.grid = document.getElementById(gridId);
    this.scenes = scenes;
    this.methodOrder = methodOrder;
    this.sceneDisplayLabels = sceneDisplayLabels;
    this.viewerMode = viewerMode;
    this.currentSceneKey = Object.keys(scenes)[0] ?? "";
    this.viewers = [];
  }

  init() {
    if (!this.grid) return false;
    if (this.viewerMode !== "grid" && !this.scenePicker) return false;

    this.createViewers();
    if (this.viewerMode !== "grid") {
      this.initScenePicker();
    }

    this.showRedInput?.addEventListener("change", () => {
      const showRed = Boolean(this.showRedInput.checked);
      for (const viewer of this.viewers) {
        viewer.setRedPointVisibility(showRed);
      }
      this.setStatus(showRed ? "Red points: ON" : "Red points: OFF");
    });

    this.resetViewBtn?.addEventListener("click", () => {
      for (const viewer of this.viewers) {
        const sceneKey = viewer.sceneKey ?? this.currentSceneKey;
        viewer.resetView(sceneKey);
      }
      if (this.viewerMode === "compare") {
        this.syncAllViewerCamerasFromFirst();
      }
      this.setStatus("View reset");
    });

    this.resize();
    return true;
  }

  createViewers() {
    this.grid.innerHTML = "";
    if (this.viewerMode === "compare") {
      const compareMethods = this.methodOrder.filter((method) => method !== "ours");
      this.viewers = compareMethods.map((method) => new CompareViewer(this.grid, method));
      return;
    }
    if (this.viewerMode === "grid") {
      this.viewers = Object.keys(this.scenes).map((sceneKey) => {
        const viewer = new SingleViewer(this.grid, this.getSceneDisplayName(sceneKey));
        viewer.sceneKey = sceneKey;
        return viewer;
      });
      return;
    }
    this.viewers = [new SingleViewer(this.grid, "Ghost-FWL raw point cloud")];
  }

  getSceneDisplayName(sceneKey) {
    return this.sceneDisplayLabels[sceneKey] ?? sceneKey;
  }

  setStatus(message) {
    if (this.statusText) {
      this.statusText.textContent = message;
    }
  }

  setActiveSceneButton(sceneKey) {
    if (!this.scenePicker) return;
    const buttons = this.scenePicker.querySelectorAll("button[data-scene]");
    for (const button of buttons) {
      const active = button.getAttribute("data-scene") === sceneKey;
      button.classList.toggle("is-active", active);
    }
  }

  createSceneThumb(sceneKey, sceneName) {
    const imagePath = `./static/images/${sceneKey}.png`;
    const img = document.createElement("img");
    img.src = imagePath;
    img.alt = sceneName;
    img.loading = "lazy";
    img.className = "pcd-scene-thumb";

    img.addEventListener("error", () => {
      const placeholder = document.createElement("div");
      placeholder.className = "pcd-scene-thumb pcd-scene-thumb-placeholder";
      placeholder.textContent = sceneName;
      img.replaceWith(placeholder);
    }, { once: true });

    return img;
  }

  initScenePicker() {
    const sceneKeys = Object.keys(this.scenes);
    if (!sceneKeys.length) {
      throw new Error("No comparison scenes are defined.");
    }

    this.scenePicker.innerHTML = "";
    if (!sceneKeys.includes(this.currentSceneKey)) {
      this.currentSceneKey = sceneKeys[0];
    }

    for (const sceneKey of sceneKeys) {
      const sceneName = this.getSceneDisplayName(sceneKey);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pcd-scene-btn";
      button.setAttribute("data-scene", sceneKey);
      button.setAttribute("aria-label", `Select scene ${sceneName}`);
      button.appendChild(this.createSceneThumb(sceneKey, sceneName));

      const label = document.createElement("span");
      label.className = "pcd-scene-label";
      label.textContent = sceneName;
      button.appendChild(label);

      button.addEventListener("click", async () => {
        if (this.currentSceneKey === sceneKey) return;
        this.currentSceneKey = sceneKey;
        this.setActiveSceneButton(sceneKey);
        try {
          await this.loadScene({ fitView: true });
        } catch (error) {
          this.setStatus(`Load failed: ${error?.message ?? String(error)}`);
          console.error(error);
        }
      });

      this.scenePicker.appendChild(button);
    }

    this.setActiveSceneButton(this.currentSceneKey);
  }

  getSceneURLs(sceneKey) {
    const sceneEntry = this.scenes[sceneKey];
    if (!sceneEntry) return null;

    const urls = {};
    for (const method of this.methodOrder) {
      const filename = sceneEntry[method];
      if (!filename) return null;
      urls[method] = `./static/pcd/${filename}`;
    }
    return urls;
  }

  syncAllViewerCamerasFromFirst() {
    if (this.viewers.length <= 1) return;
    const source = this.viewers[0];
    for (let i = 1; i < this.viewers.length; i += 1) {
      this.viewers[i].copyCameraFrom(source);
    }
  }

  resize() {
    for (const viewer of this.viewers) {
      viewer.resize();
    }
  }

  render() {
    for (const viewer of this.viewers) {
      viewer.render();
    }
  }

  async loadScene({ fitView = false } = {}) {
    if (this.viewerMode === "grid") {
      return this.loadAllScenes({ fitView });
    }

    const sceneKey = this.currentSceneKey || Object.keys(this.scenes)[0];
    if (!sceneKey) return false;
    this.currentSceneKey = sceneKey;
    this.setActiveSceneButton(sceneKey);

    const sceneName = this.getSceneDisplayName(sceneKey);
    const urls = this.getSceneURLs(sceneKey);
    if (!urls) return false;

    this.setStatus(`Loading: ${sceneName}`);
    const showRed = Boolean(this.showRedInput?.checked);

    const entries = await Promise.all(
      this.methodOrder.map(async (method) => {
        const points = await loadPCDFromURL(urls[method]);
        return [method, points];
      }),
    );
    const pointsByMethod = Object.fromEntries(entries);

    if (this.viewerMode === "compare") {
      for (let i = 0; i < this.viewers.length; i += 1) {
        const viewer = this.viewers[i];
        const shouldFit = fitView && i === 0;
        viewer.setComparisonPoints(pointsByMethod[viewer.compareMethod], pointsByMethod.ours, {
          fitView: shouldFit,
          sceneKey,
          showRed,
        });
      }
      if (fitView) {
        this.syncAllViewerCamerasFromFirst();
      }
      this.setStatus(`Loaded: ${sceneName} / all baselines vs ours`);
      return true;
    }

    this.viewers[0]?.setPoints(pointsByMethod["ghost-fwl"], {
      fitView,
      sceneKey,
      showRed,
    });
    this.setStatus(`Loaded: ${sceneName}`);
    return true;
  }

  async loadAllScenes({ fitView = false } = {}) {
    const sceneKeys = Object.keys(this.scenes);
    if (!sceneKeys.length) return false;

    this.setStatus(`Loading: ${sceneKeys.length} scenes`);
    const showRed = Boolean(this.showRedInput?.checked);

    const entries = await Promise.all(
      sceneKeys.map(async (sceneKey) => {
        const urls = this.getSceneURLs(sceneKey);
        if (!urls) {
          throw new Error(`missing scene entry: ${sceneKey}`);
        }
        const points = await loadPCDFromURL(urls["ghost-fwl"]);
        return [sceneKey, points];
      }),
    );

    const pointsByScene = Object.fromEntries(entries);
    for (const viewer of this.viewers) {
      const sceneKey = viewer.sceneKey;
      const points = pointsByScene[sceneKey];
      viewer.setPoints(points, {
        fitView,
        sceneKey,
        showRed,
      });
    }

    this.setStatus(`Loaded: ${sceneKeys.length} dataset scenes`);
    return true;
  }

  async tryInitialLoad() {
    const ok = await this.loadScene({ fitView: true });
    if (!ok) {
      this.setStatus("Initial load failed");
    }
  }
}

const apps = [
  new PcdApp({
    scenePickerId: "datasetPcdScenePicker",
    showRedInputId: "datasetPcdShowRedPoints",
    resetViewBtnId: "datasetPcdResetViewBtn",
    statusId: "datasetPcdStatus",
    gridId: "datasetPcdCompareGrid",
    scenes: DATASET_SCENES,
    methodOrder: DATASET_METHOD_ORDER,
    sceneDisplayLabels: DATASET_SCENE_DISPLAY_LABELS,
    viewerMode: "grid",
  }),
  new PcdApp({
    scenePickerId: "pcdScenePicker",
    showRedInputId: "pcdShowRedPoints",
    resetViewBtnId: "pcdResetViewBtn",
    statusId: "pcdStatus",
    gridId: "pcdCompareGrid",
    scenes: RESULT_SCENES,
    methodOrder: RESULT_METHOD_ORDER,
    sceneDisplayLabels: RESULT_SCENE_DISPLAY_LABELS,
    viewerMode: "compare",
  }),
].filter((app) => app.init());

function resizeAll() {
  for (const app of apps) {
    app.resize();
  }
}

window.addEventListener("resize", resizeAll);
resizeAll();

function animate() {
  requestAnimationFrame(animate);
  for (const app of apps) {
    app.render();
  }
}

animate();

for (const app of apps) {
  app.tryInitialLoad();
}
