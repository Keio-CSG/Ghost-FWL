import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";

const scenePicker = document.getElementById("pcdScenePicker");
const showRedPointsInput = document.getElementById("pcdShowRedPoints");
const resetViewBtn = document.getElementById("pcdResetViewBtn");
const statusText = document.getElementById("pcdStatus");
const compareGrid = document.getElementById("pcdCompareGrid");

if (!scenePicker || !compareGrid) {
  throw new Error("PCD compare UI is missing.");
}

const loader = new PCDLoader();
const pointsCache = new Map();

const METHOD_ORDER = ["GT", "marmot", "lindell", "ours"];
const BASELINE_METHODS = METHOD_ORDER.filter((m) => m !== "ours");
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
  "22build": {
    distScale: 0.34,
    distMin: 0.38,
    leftScale: 0.58,
    leftMin: 0.54,
    downScale: 0.06,
    downMin: 0.10,
    targetDownScale: 0.12,
    targetDownMin: 0.16,
    targetShiftLeftScale: 0.0,
    targetShiftLeftMin: 0.0,
    yawLeftDeg: 0,
    zoomMul: 3.4,
    fovDeg: 34,
    cameraZoom: 1.5,
    pointYawLeftDeg: 48,
    pointShiftRight: 0.55,
    pointShiftToward: 5.56,
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
// Display-axis correction matrix (identity).
const DISPLAY_AXIS_CORRECTION = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
);
const COMPARISON_SCENES = {
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
const SCENE_DISPLAY_LABELS = {
  "14build_7floor": "Scene001",
  "22build": "Scene007",
  "36build": "Scene009",
};
let currentSceneKey = Object.keys(COMPARISON_SCENES)[0] ?? "";

function setStatus(message) {
  if (statusText) statusText.textContent = message;
}

function setActiveSceneButton(sceneKey) {
  const buttons = scenePicker.querySelectorAll("button[data-scene]");
  for (const button of buttons) {
    const active = button.getAttribute("data-scene") === sceneKey;
    button.classList.toggle("is-active", active);
  }
}

function initScenePicker() {
  const sceneKeys = Object.keys(COMPARISON_SCENES);
  if (!sceneKeys.length) {
    throw new Error("No comparison scenes are defined.");
  }

  scenePicker.innerHTML = "";
  if (!sceneKeys.includes(currentSceneKey)) {
    currentSceneKey = sceneKeys[0];
  }

  for (const sceneKey of sceneKeys) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pcd-scene-btn";
    button.setAttribute("data-scene", sceneKey);
    button.setAttribute("aria-label", `Select scene ${sceneKey}`);

    const img = document.createElement("img");
    img.src = `./static/images/${sceneKey}.png`;
    img.alt = sceneKey;
    img.loading = "lazy";
    img.className = "pcd-scene-thumb";
    button.appendChild(img);

    const label = document.createElement("span");
    label.className = "pcd-scene-label";
    label.textContent = SCENE_DISPLAY_LABELS[sceneKey] ?? sceneKey;
    button.appendChild(label);

    button.addEventListener("click", async () => {
      if (currentSceneKey === sceneKey) return;
      currentSceneKey = sceneKey;
      setActiveSceneButton(sceneKey);
      try {
        await loadSceneComparisons({ fitView: true });
      } catch (error) {
        setStatus(`Load failed: ${error?.message ?? String(error)}`);
        console.error(error);
      }
    });

    scenePicker.appendChild(button);
  }

  setActiveSceneButton(currentSceneKey);
}

function getCameraPreset(sceneKey) {
  return CAMERA_PRESETS[sceneKey] ?? CAMERA_PRESETS.default;
}

function fitCameraToBox(camera, controls, box, sceneKey = currentSceneKey) {
  if (box.isEmpty()) {
    return;
  }

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
  // Axis-aligned view:
  // depth(away)=+X, screen right=+Y, screen down=+Z
  controls.target.copy(center).add(new THREE.Vector3(0, -targetShiftLeftOffset, targetDownOffset));
  camera.up.set(0, 0, -1);
  const relativePosition = new THREE.Vector3(-dist, -leftOffset, downOffset);
  if (yawLeftRad !== 0) {
    relativePosition.applyAxisAngle(camera.up, yawLeftRad);
  }
  camera.position.copy(center).add(relativePosition);
  controls.update();
}

function fitCameraToPair(camera, controls, leftPoints, rightPoints, sceneKey = currentSceneKey) {
  const box = new THREE.Box3();
  const hasLeft = Boolean(leftPoints);
  const hasRight = Boolean(rightPoints);
  if (!hasLeft && !hasRight) return;

  if (hasLeft) {
    box.union(new THREE.Box3().setFromObject(leftPoints));
  }
  if (hasRight) {
    box.union(new THREE.Box3().setFromObject(rightPoints));
  }
  if (box.isEmpty()) return;
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

function buildNonRedGeometry(geometry) {
  const colorAttr = geometry?.getAttribute?.("color");
  const positionAttr = geometry?.getAttribute?.("position");
  const count = positionAttr?.count ?? 0;
  if (!colorAttr || !positionAttr || count === 0) return null;

  const kept = [];
  for (let i = 0; i < count; i += 1) {
    const r = colorAttr.getX(i);
    const g = colorAttr.getY(i);
    const b = colorAttr.getZ(i);
    if (!isRedPoint(r, g, b)) kept.push(i);
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

function applyRedPointVisibility(points) {
  const showRed = Boolean(showRedPointsInput?.checked);
  const fullGeometry = points.userData?.fullGeometry;
  const nonRedGeometry = points.userData?.nonRedGeometry;
  if (!fullGeometry) return;

  if (showRed || !nonRedGeometry) {
    if (points.geometry !== fullGeometry) {
      points.geometry = fullGeometry;
    }
    return;
  }
  if (points.geometry !== nonRedGeometry) {
    points.geometry = nonRedGeometry;
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

function clonePoints(points, sceneKey = currentSceneKey) {
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
  cloned.name = points.name;
  cloned.userData.fullGeometry = geometry;
  cloned.userData.nonRedGeometry = buildNonRedGeometry(geometry);
  applyRedPointVisibility(cloned);
  return cloned;
}

function applyPointTranslation(points, sceneKey = currentSceneKey) {
  if (!points) return;
  const preset = getCameraPreset(sceneKey);
  const base = points.userData?.basePosition ?? new THREE.Vector3(0, 0, 0);
  const pointShiftRight = preset.pointShiftRight ?? 0;
  const pointShiftToward = preset.pointShiftToward ?? 0;
  points.position.copy(base);
  points.position.y += pointShiftRight;
  // In this view setup, +X is depth-away, so toward camera is -X.
  points.position.x -= pointShiftToward;
}

function disposePoints(points) {
  if (!points) return;
  points.userData?.fullGeometry?.dispose?.();
  if (points.userData?.nonRedGeometry && points.userData.nonRedGeometry !== points.userData.fullGeometry) {
    points.userData.nonRedGeometry.dispose();
  }
  points.material?.dispose?.();
}

class CompareViewer {
  constructor(compareMethod) {
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

    compareGrid.appendChild(this.root);
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

  setRedPointVisibility() {
    if (this.leftPoints) applyRedPointVisibility(this.leftPoints);
    if (this.rightPoints) applyRedPointVisibility(this.rightPoints);
  }

  setComparisonPoints(leftRaw, rightRaw, { fitView = false, sceneKey = currentSceneKey } = {}) {
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

    this.leftScene.add(this.leftPoints);
    this.rightScene.add(this.rightPoints);
    this.setPointSize();

    if (fitView) {
      fitCameraToPair(this.camera, this.controls, this.leftPoints, this.rightPoints, sceneKey);
    }
    applyPointTranslation(this.leftPoints, sceneKey);
    applyPointTranslation(this.rightPoints, sceneKey);
  }

  resetView(sceneKey = currentSceneKey) {
    fitCameraToPair(this.camera, this.controls, this.leftPoints, this.rightPoints, sceneKey);
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

    // Render each cloud with full-canvas projection, and reveal by scissor only.
    // This keeps both methods "spread" over the entire view while slider switches visibility.
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

const viewers = BASELINE_METHODS.map((method) => new CompareViewer(method));

function syncAllViewerCamerasFromFirst() {
  if (viewers.length <= 1) return;
  const source = viewers[0];
  for (let i = 1; i < viewers.length; i += 1) {
    viewers[i].copyCameraFrom(source);
  }
}

function getSceneURLs(sceneKey) {
  const sceneEntry = COMPARISON_SCENES[sceneKey];
  if (!sceneEntry) return null;

  const urls = {};
  for (const method of METHOD_ORDER) {
    const filename = sceneEntry[method];
    if (!filename) return null;
    urls[method] = `./static/pcd/${filename}`;
  }
  return urls;
}

showRedPointsInput?.addEventListener("change", () => {
  for (const viewer of viewers) {
    viewer.setRedPointVisibility();
  }
  setStatus(showRedPointsInput.checked ? "Red points: ON" : "Red points: OFF");
});

resetViewBtn?.addEventListener("click", () => {
  if (viewers[0]) {
    viewers[0].resetView(currentSceneKey);
    syncAllViewerCamerasFromFirst();
  }
  setStatus("View reset");
});

function resize() {
  for (const viewer of viewers) {
    viewer.resize();
  }
}

window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  for (const viewer of viewers) {
    viewer.render();
  }
}
animate();

async function loadSceneComparisons({ fitView = false } = {}) {
  const sceneKey = currentSceneKey || Object.keys(COMPARISON_SCENES)[0];
  if (!sceneKey) return false;
  currentSceneKey = sceneKey;
  setActiveSceneButton(sceneKey);
  const urls = getSceneURLs(sceneKey);
  if (!urls) return false;

  setStatus(`Loading: ${sceneKey}`);
  const entries = await Promise.all(
    METHOD_ORDER.map(async (method) => {
      const points = await loadPCDFromURL(urls[method]);
      return [method, points];
    }),
  );

  const pointsByMethod = Object.fromEntries(entries);
  for (let i = 0; i < viewers.length; i += 1) {
    const viewer = viewers[i];
    const shouldFit = fitView && i === 0;
    viewer.setComparisonPoints(pointsByMethod[viewer.compareMethod], pointsByMethod.ours, {
      fitView: shouldFit,
      sceneKey,
    });
  }
  if (fitView) {
    syncAllViewerCamerasFromFirst();
  }

  setStatus(`Loaded: ${sceneKey} / all baselines vs ours`);
  return true;
}

async function tryInitialLoad() {
  const ok = await loadSceneComparisons({ fitView: true });
  if (ok) return;
  setStatus("Initial load failed");
}

initScenePicker();
tryInitialLoad();
