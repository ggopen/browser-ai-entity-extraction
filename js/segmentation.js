// AI segmentation wrapper using ONNX Runtime Web.
// Loads a U2Net-P salient object detection model and produces a binary
// foreground mask per input image. Connected components over the mask
// yield instance ids.

import { logger } from './logger.js';

const MODEL_PATH = './models/u2netp.onnx';
const MODEL_SIZE_BYTES = 4574861; // ~4.4 MB
const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

let _ortPromise = null;
function loadOrt() {
  if (_ortPromise) return _ortPromise;
  _ortPromise = import('onnxruntime-web').then((mod) => {
    if (mod.env) {
      // Use the CDN-hosted WASM files for the WASM backend.
      mod.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
      // Multi-threaded WASM requires SharedArrayBuffer and cross-origin
      // isolation. GitHub Pages does not set COOP/COEP headers, so the
      // threaded worker would silently fail. Force single-threaded mode
      // to keep startup deterministic in non-isolated contexts.
      mod.env.wasm.numThreads = 1;
      mod.env.wasm.simd = true;
    }
    return mod;
  });
  return _ortPromise;
}

let _session = null;
let _loading = null;

export async function getSegmentationModel(progressCb) {
  if (_session) return _session;
  if (_loading) return _loading;
  _loading = (async () => {
    const ort = await loadOrt();
    // Choose execution providers. WebGPU first if available, then WebGL, then WASM.
    const providers = [];
    // Note: WebGL EP is not always stable for arbitrary models; prefer WASM by default
    if (ort.env && ort.env.wasm) {
      providers.push('wasm');
    }
    const opts = {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      executionMode: 'sequential'
    };
    logger.info('正在加载 AI 分割模型...');
    try {
      _session = await ort.InferenceSession.create(MODEL_PATH, opts);
    } catch (err) {
      logger.warn(`默认 EP 加载失败，重试 wasm: ${err.message}`);
      _session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
    }
    logger.ok(`AI 模型加载完成 (输入: ${JSON.stringify(_session.inputNames)}, 输出: ${JSON.stringify(_session.outputNames)})`);
    return _session;
  })();
  return _loading;
}

export function getModelSize() {
  return MODEL_SIZE_BYTES;
}

// Pre-process: take an RGBA Uint8ClampedArray of size w*h*4, return
// a Float32Array of shape [1, 3, 320, 320] (CHW normalized) suitable
// as the model input.
export function preprocess(rgba, w, h) {
  // Resize by nearest neighbor to 320x320 (we keep this lightweight).
  const out = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  const scaleX = w / INPUT_SIZE;
  const scaleY = h / INPUT_SIZE;

  for (let y = 0; y < INPUT_SIZE; y++) {
    const sy = Math.min(h - 1, Math.floor(y * scaleY));
    for (let x = 0; x < INPUT_SIZE; x++) {
      const sx = Math.min(w - 1, Math.floor(x * scaleX));
      const i = (sy * w + sx) * 4;
      const r = rgba[i]     / 255.0;
      const g = rgba[i + 1] / 255.0;
      const b = rgba[i + 2] / 255.0;
      const idx = y * INPUT_SIZE + x;
      out[0 * INPUT_SIZE * INPUT_SIZE + idx] = (r - MEAN[0]) / STD[0];
      out[1 * INPUT_SIZE * INPUT_SIZE + idx] = (g - MEAN[1]) / STD[1];
      out[2 * INPUT_SIZE * INPUT_SIZE + idx] = (b - MEAN[2]) / STD[2];
    }
  }
  return out;
}

// Post-process: take the model output (Float32Array 320*320) and return
// a binary mask of size w*h (Uint8Array, 0/1) resized to the target.
export function postprocessMask(out, targetW, targetH, threshold = 0.5) {
  const mask = new Uint8Array(targetW * targetH);
  const scaleX = INPUT_SIZE / targetW;
  const scaleY = INPUT_SIZE / targetH;
  for (let y = 0; y < targetH; y++) {
    const sy = Math.min(INPUT_SIZE - 1, Math.floor(y * scaleY));
    for (let x = 0; x < targetW; x++) {
      const sx = Math.min(INPUT_SIZE - 1, Math.floor(x * scaleX));
      const v = out[sy * INPUT_SIZE + sx];
      mask[y * targetW + x] = v >= threshold ? 1 : 0;
    }
  }
  return mask;
}

// Run inference on a single image, return binary mask of size w*h.
export async function inferOne(rgba, w, h) {
  const session = await getSegmentationModel();
  const ort = await loadOrt();
  const input = preprocess(rgba, w, h);
  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  // Main output is '1959' or the first listed output
  const outName = session.outputNames[0];
  const outArr = results[outName].data;
  return postprocessMask(outArr, w, h);
}

// Connected-components on a binary mask. Returns:
//   { labels: Int32Array of label per pixel, count: number of components }
export function connectedComponents(mask, w, h) {
  const labels = new Int32Array(w * h).fill(0);
  const queueX = new Int32Array(w * h);
  const queueY = new Int32Array(w * h);
  let label = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (mask[i] === 0 || labels[i] !== 0) continue;
      label++;
      let head = 0, tail = 0;
      queueX[tail] = x;
      queueY[tail] = y;
      tail++;
      labels[i] = label;
      while (head < tail) {
        const cx = queueX[head];
        const cy = queueY[head];
        head++;
        // 4-neighborhood
        if (cx > 0)     { const j = cy * w + (cx - 1); if (mask[j] && !labels[j]) { labels[j] = label; queueX[tail] = cx - 1; queueY[tail] = cy; tail++; } }
        if (cx < w - 1) { const j = cy * w + (cx + 1); if (mask[j] && !labels[j]) { labels[j] = label; queueX[tail] = cx + 1; queueY[tail] = cy; tail++; } }
        if (cy > 0)     { const j = (cy - 1) * w + cx; if (mask[j] && !labels[j]) { labels[j] = label; queueX[tail] = cx; queueY[tail] = cy - 1; tail++; } }
        if (cy < h - 1) { const j = (cy + 1) * w + cx; if (mask[j] && !labels[j]) { labels[j] = label; queueX[tail] = cx; queueY[tail] = cy + 1; tail++; } }
      }
    }
  }
  return { labels, count: label };
}

// Remove small components (size < minSize) by setting them to 0.
export function filterSmallComponents(labels, count, minSize) {
  const sizes = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) {
    sizes[labels[i]]++;
  }
  const filtered = new Int32Array(labels.length);
  let newId = 0;
  const remap = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) {
    const old = labels[i];
    if (old === 0 || sizes[old] < minSize) {
      filtered[i] = 0;
      continue;
    }
    if (remap[old] === 0) {
      newId++;
      remap[old] = newId;
    }
    filtered[i] = remap[old];
  }
  return { labels: filtered, count: newId };
}

// Run segmentation on multiple images, returning per-image instance labels.
export async function inferBatch(images) {
  const session = await getSegmentationModel();
  const ort = await loadOrt();
  const out = [];
  for (const img of images) {
    const input = preprocess(img.rgb, img.width, img.height);
    const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds = { [session.inputNames[0]]: tensor };
    const t0 = performance.now();
    const results = await session.run(feeds);
    const dt = performance.now() - t0;
    const outArr = results[session.outputNames[0]].data;
    const mask = postprocessMask(outArr, img.width, img.height, 0.5);
    const cc = connectedComponents(mask, img.width, img.height);
    const filtered = filterSmallComponents(cc.labels, cc.count, 64);
    out.push({
      name: img.name,
      camera: img.camera,
      width: img.width,
      height: img.height,
      labels: filtered.labels,
      count: filtered.count,
      inferenceMs: dt
    });
  }
  return out;
}
