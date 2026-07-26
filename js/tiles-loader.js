// Minimal 3D Tiles + b3dm loader for Three.js. Produces Three.js Object3D
// groups with world transforms already applied to the geometry (so the
// renderer only needs to do identity transforms).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { logger } from './logger.js';

const B3DM_MAGIC = 0x6d643362; // 'b3dm' little-endian
const I3DM_MAGIC = 0x6d646933;
const PNTS_MAGIC = 0x73746e70;

function readUInt32LE(view, off) {
  return view.getUint32(off, true);
}

function parseB3dmHeader(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const magic = readUInt32LE(view, 0);
  if (magic !== B3DM_MAGIC) {
    throw new Error(`Not a b3dm file (magic=${magic.toString(16)})`);
  }
  const version = readUInt32LE(view, 4);
  const byteLength = readUInt32LE(view, 8);
  const ftJsonLen = readUInt32LE(view, 12);
  const ftBinLen = readUInt32LE(view, 16);
  const btJsonLen = readUInt32LE(view, 20);
  const btBinLen = readUInt32LE(view, 24);
  const glbOffset = 28 + ftJsonLen + ftBinLen + btJsonLen + btBinLen;
  const header = {
    version,
    byteLength,
    ftJsonLen,
    ftBinLen,
    btJsonLen,
    btBinLen,
    glbOffset,
    featureTable: null,
    batchTable: null
  };
  if (ftJsonLen > 0) {
    // The JSON is padded with spaces / nulls to 4 bytes; trim them
    // off before parsing. We treat anything <= 0x20 (whitespace and
    // null bytes) as padding.
    const raw = new Uint8Array(arrayBuffer, 28, ftJsonLen);
    let end = ftJsonLen;
    while (end > 0) {
      const b = raw[end - 1];
      if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d || b === 0x00) end--;
      else break;
    }
    const text = new TextDecoder('utf-8').decode(raw.subarray(0, end));
    header.featureTable = JSON.parse(text);
  }
  if (btJsonLen > 0) {
    const raw = new Uint8Array(arrayBuffer, 28 + ftJsonLen + ftBinLen, btJsonLen);
    let end = btJsonLen;
    while (end > 0) {
      const b = raw[end - 1];
      if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d || b === 0x00) end--;
      else break;
    }
    const text = new TextDecoder('utf-8').decode(raw.subarray(0, end));
    header.batchTable = JSON.parse(text);
  }
  return header;
}

function extractGlb(arrayBuffer, glbOffset) {
  return arrayBuffer.slice(glbOffset);
}

function buildRtcTransform(featureTable) {
  if (featureTable && featureTable.RTC_CENTER) {
    return new THREE.Matrix4().makeTranslation(
      featureTable.RTC_CENTER[0],
      featureTable.RTC_CENTER[1],
      featureTable.RTC_CENTER[2]
    );
  }
  return new THREE.Matrix4();
}

const COMPONENT_TYPE = {
  5120: Int8Array,
  5121: Uint8Array,
  5122: Int16Array,
  5123: Uint16Array,
  5125: Uint32Array,
  5126: Float32Array
};
const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_SIZE = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

// Parse a glTF 1.0 / 2.0 binary file (GLB). Returns a Three.js Group
// with one Mesh per primitive. This is used because the embedded GLB
// inside legacy b3dm 1.0 tiles is glTF 1.0 binary, which three.js's
// built-in GLTFLoader no longer supports.
function parseGlbBinary(glbBuf) {
  const dv = new DataView(glbBuf);
  const magic = dv.getUint32(0, true);
  const GLB_MAGIC = 0x46546C67; // 'glTF'
  if (magic !== GLB_MAGIC) {
    throw new Error(`Not a GLB file (magic=${magic.toString(16)})`);
  }
  const version = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);

  let json = null;
  let binOffset = -1;
  let binLength = 0;

  if (version < 2) {
    // glTF 1.0 binary format used by KHR_binary_glTF tiles in b3dm:
    //   12-byte header (magic, version, length)
    //   4-byte JSON chunk length
    //   4-byte BIN chunk length (0 means "BIN extends to end of file")
    //   JSON data (NOT padded to 4-byte boundary in this format)
    //   BIN data (NOT padded; extends to the end if the BIN length was 0)
    // The BIN data holds buffers referenced by the JSON via the
    // bufferView byteOffsets.
    const jsonLen = dv.getUint32(12, true);
    const declaredBinLen = dv.getUint32(16, true);
    // JSON data starts at offset 20 (after the 4-byte JSON length and
    // 4-byte BIN length). It is NOT padded to a 4-byte boundary.
    const jsonDataOff = 20;
    if (jsonDataOff + jsonLen > glbBuf.byteLength) {
      throw new Error(`glTF 1.0 JSON 长度越界: ${jsonLen}`);
    }
    const raw = new Uint8Array(glbBuf, jsonDataOff, jsonLen);
    let end = jsonLen;
    while (end > 0) {
      const b = raw[end - 1];
      if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d || b === 0x00) end--;
      else break;
    }
    const text = new TextDecoder('utf-8').decode(raw.subarray(0, end));
    try {
      json = JSON.parse(text);
    } catch (e) {
      logger.error(`GLB JSON 解析失败: ${e.message}`);
      const m = /at position (\d+)/.exec(e.message);
      if (m) {
        const pos = parseInt(m[1], 10);
        logger.error(`附近内容: ${JSON.stringify(text.slice(Math.max(0, pos - 20), pos + 20))}`);
      }
      throw e;
    }
    // JSON is NOT padded to 4-byte boundary in this format; BIN starts
    // immediately after the last byte of JSON. The BIN offset within
    // the file may not be 4-byte aligned, so we copy it into a fresh
    // ArrayBuffer for safe TypedArray construction.
    const srcBinOffset = jsonDataOff + jsonLen;
    const srcBinLength = declaredBinLen > 0
      ? declaredBinLen
      : Math.max(0, length - srcBinOffset);
    if (srcBinLength <= 0) {
      // No BIN data; group will be empty.
      return new THREE.Group();
    }
    // Copy into a new, properly-aligned ArrayBuffer.
    const binCopy = new Uint8Array(srcBinLength);
    binCopy.set(new Uint8Array(glbBuf, srcBinOffset, srcBinLength));
    binOffset = 0;
    binLength = srcBinLength;
    // Replace glbBuf with a buffer that starts at the BIN copy. The
    // accessors reference byte offsets relative to the BIN, so an
    // ArrayBuffer that starts at the BIN keeps the existing offset
    // arithmetic correct.
    glbBuf = binCopy.buffer;
  } else {
    // glTF 2.0 binary format with typed chunks:
    //   12-byte header (magic, version, length)
    //   then 0..N chunks, each with an 8-byte header (length, type) and
    //   data padded to 4-byte boundary. Type is 'JSON' (0x4E4F534A) or
    //   'BIN\0' (0x004E4942).
    let offset = 12;
    while (offset < length) {
      if (offset + 8 > length) break;
      const chunkLen = dv.getUint32(offset, true); offset += 4;
      const chunkType = dv.getUint32(offset, true); offset += 4;
      if (chunkType === 0x4E4F534A) { // 'JSON'
        if (offset + chunkLen > glbBuf.byteLength) {
          throw new Error(`glTF 2.0 JSON 长度越界: ${chunkLen}`);
        }
        const raw = new Uint8Array(glbBuf, offset, chunkLen);
        let end = chunkLen;
        while (end > 0) {
          const b = raw[end - 1];
          if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d || b === 0x00) end--;
          else break;
        }
        const text = new TextDecoder('utf-8').decode(raw.subarray(0, end));
        try {
          json = JSON.parse(text);
        } catch (e) {
          logger.error(`GLB JSON 解析失败: ${e.message}`);
          throw e;
        }
      } else if (chunkType === 0x004E4942) { // 'BIN\0'
        binOffset = offset;
        binLength = chunkLen;
      }
      offset += chunkLen + ((4 - (chunkLen % 4)) % 4);
    }
  }
  if (!json) throw new Error('GLB has no JSON chunk');

  const group = new THREE.Group();
  if (binOffset < 0) return group;

  // glTF 1.0 represents its "JSON objects" (e.g. meshes, accessors,
  // bufferViews) as arrays of string keys, with the actual data stored
  // at the top level of the JSON under those keys. Detect this format
  // and look data up via json[keyName] rather than json.array[i].
  const isDict = (node) => Array.isArray(node) && node.length > 0
    && node.every((x) => typeof x === 'string' && json[x] !== undefined);

  const meshKeys = isDict(json.meshes) ? json.meshes : null;
  const accessorKeys = isDict(json.accessors) ? json.accessors : null;
  const bufferViewKeys = isDict(json.bufferViews) ? json.bufferViews : null;

  // Resolve a bufferView reference (either a string key for glTF 1.0
  // dict form, or a numeric index for glTF 2.0 array form).
  const resolveBufferView = (ref) => {
    if (typeof ref === 'number') return json.bufferViews?.[ref];
    if (typeof ref === 'string') {
      if (bufferViewKeys && bufferViewKeys.includes(ref)) return json[ref];
      return json.bufferViews?.[ref];
    }
    return undefined;
  };

  // Resolve an accessor reference similarly.
  const resolveAccessor = (ref) => {
    if (typeof ref === 'number') return json.accessors?.[ref];
    if (typeof ref === 'string') {
      if (accessorKeys && accessorKeys.includes(ref)) return json[ref];
      return json.accessors?.[ref];
    }
    return undefined;
  };

  // Helper: read accessor as a typed array view into the BIN data
  function readAccessor(acc) {
    const bv = resolveBufferView(acc.bufferView);
    if (!bv) throw new Error(`Accessor 引用了未知的 bufferView: ${acc.bufferView}`);
    const compT = acc.componentType;
    const TypedArray = COMPONENT_TYPE[compT];
    if (!TypedArray) throw new Error(`未知的 componentType: ${compT}`);
    const typeCount = TYPE_SIZE[acc.type];
    if (!typeCount) throw new Error(`未知的 accessor type: ${acc.type}`);
    const compBytes = COMPONENT_BYTES[compT];
    const total = acc.count * typeCount;
    // Sanity-check: refuse obviously bogus values that would crash
    // TypedArray construction or read past the BIN chunk.
    const byteLen = total * compBytes;
    if (!Number.isFinite(total) || total < 0 || byteLen > binLength) {
      throw new Error(
        `Accessor 越界: count=${acc.count}, type=${acc.type}, ` +
        `byteLength=${byteLen}, binLength=${binLength}`
      );
    }
    const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
    if (byteOffset + byteLen > binLength) {
      throw new Error(
        `Accessor 越界: byteOffset=${byteOffset}, byteLength=${byteLen}, ` +
        `binLength=${binLength}`
      );
    }
    return new TypedArray(glbBuf, binOffset + byteOffset, total);
  }

  // For each mesh primitive, build a Three.js BufferGeometry
  const meshList = meshKeys
    ? meshKeys.map((k) => json[k]).filter(Boolean)
    : (Array.isArray(json.meshes) ? json.meshes : []);
  for (const mesh of meshList) {
    for (const prim of mesh.primitives || []) {
      try {
        const g = new THREE.BufferGeometry();
        // Map glTF attribute semantics to Three.js names
        const SEMANTIC_MAP = {
          POSITION: 'position',
          NORMAL: 'normal',
          TANGENT: 'tangent',
          TEXCOORD_0: 'uv',
          TEXCOORD_1: 'uv1',
          TEXCOORD_2: 'uv2',
          TEXCOORD_3: 'uv3',
          COLOR_0: 'color',
          JOINTS_0: 'skinIndex',
          WEIGHTS_0: 'skinWeight'
        };
        // Attributes
        const attrs = prim.attributes || {};
        for (const [name, accRef] of Object.entries(attrs)) {
          const acc = resolveAccessor(accRef);
          if (!acc) {
            logger.warn(`属性 ${name} 引用了未知的 accessor: ${accRef}`);
            continue;
          }
          const arr = readAccessor(acc);
          const itemSize = TYPE_SIZE[acc.type];
          const threeName = SEMANTIC_MAP[name] || name;
          g.setAttribute(threeName, new THREE.BufferAttribute(arr, itemSize));
        }
        // Indices
        if (prim.indices !== undefined) {
          const acc = resolveAccessor(prim.indices);
          if (!acc) {
            logger.warn(`索引引用了未知的 accessor: ${prim.indices}`);
            continue;
          }
          const arr = readAccessor(acc);
          const itemSize = TYPE_SIZE[acc.type];
          g.setIndex(new THREE.BufferAttribute(arr, itemSize));
        }
        // Compute normals if missing
        if (!g.attributes.normal) {
          g.computeVertexNormals();
        }
        // Default material (no PBR; the b3dm contains baked colors)
        const mat = new THREE.MeshBasicMaterial({ vertexColors: !!g.attributes.color, side: THREE.FrontSide });
        const meshObj = new THREE.Mesh(g, mat);
        group.add(meshObj);
      } catch (e) {
        console.warn('Failed to build primitive', e);
      }
    }
  }
  return group;
}

const gltfLoader = new GLTFLoader();
const gltfCache = new Map();

// fetch with retry for transient 5xx / network errors. The remote tile
// service at data.mars3d.cn frequently returns 503 under load; we back
// off and retry a couple of times before giving up.
async function fetchWithRetry(url, attempts = 3) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return r;
      // Retry on transient 5xx and 429
      if (r.status >= 500 || r.status === 429) {
        lastErr = new Error(`HTTP ${r.status} for ${url}`);
        if (i < attempts - 1) {
          const wait = 400 * Math.pow(2, i) + Math.random() * 200;
          logger.warn(`瓦片 ${url.split('/').pop()} 返回 ${r.status}，${wait.toFixed(0)} ms 后重试 (${i + 1}/${attempts})`);
          await new Promise((res) => setTimeout(res, wait));
          continue;
        }
      }
      throw new Error(`HTTP ${r.status} for ${url}`);
    } catch (e) {
      lastErr = e;
      if (i < attempts - 1) {
        const wait = 400 * Math.pow(2, i) + Math.random() * 200;
        logger.warn(`瓦片 ${url.split('/').pop()} 请求失败: ${e.message}，${wait.toFixed(0)} ms 后重试 (${i + 1}/${attempts})`);
        await new Promise((res) => setTimeout(res, wait));
      }
    }
  }
  throw lastErr;
}

function loadGlb(url) {
  if (gltfCache.has(url)) return gltfCache.get(url);
  const promise = fetchWithRetry(url)
    .then((r) => r.arrayBuffer())
    .then((buf) => {
      const header = parseB3dmHeader(buf);
      const glbBuf = extractGlb(buf, header.glbOffset);
      const rtc = buildRtcTransform(header.featureTable);

      // Check the GLB version. If it's glTF 1.0 binary, skip the
      // GLTFLoader (which doesn't support it) and use our custom
      // parser directly. If it's glTF 2.0, use GLTFLoader.
      const glbMagic = new DataView(glbBuf).getUint32(0, true);
      const glbVersion = new DataView(glbBuf).getUint32(4, true);
      if (glbMagic !== 0x46546C67) {
        // Not a GLB - try GLTFLoader anyway (handles plain glTF JSON)
        return runGltfLoader(glbBuf, rtc);
      }
      if (glbVersion < 2) {
        // glTF 1.0 binary: use our custom parser
        const group = parseGlbBinary(glbBuf);
        group.traverse((o) => {
          if (o.isMesh && o.geometry) o.geometry.applyMatrix4(rtc);
        });
        return { scene: group };
      }
      // glTF 2.0 binary: use GLTFLoader
      return runGltfLoader(glbBuf, rtc);
    });
  gltfCache.set(url, promise);
  return promise;
}

function runGltfLoader(glbBuf, rtc) {
  return new Promise((resolve, reject) => {
    gltfLoader.parse(
      glbBuf,
      '',
      (gltf) => {
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.geometry) o.geometry.applyMatrix4(rtc);
        });
        resolve(gltf);
      },
      (err) => reject(new Error('GLTFLoader: ' + (err?.message || err)))
    );
  });
}

// Build a Three.js Box3 in ECEF coords for a 3D Tiles boundingVolume.
function makeBoundingBox3(bv) {
  const box = new THREE.Box3();
  if (!bv) return box.makeEmpty();
  if (bv.box) {
    // [centerX, centerY, centerZ, xAxisLen, yAxisLen, zAxisLen, ...]
    // We treat as an axis-aligned box in ECEF.
    const c = bv.box;
    box.min.set(c[0] - c[3], c[1] - c[4], c[2] - c[5]);
    box.max.set(c[0] + c[3], c[1] + c[4], c[2] + c[5]);
  } else if (bv.sphere) {
    const s = bv.sphere;
    const r = s[3];
    box.min.set(s[0] - r, s[1] - r, s[2] - r);
    box.max.set(s[0] + r, s[1] + r, s[2] + r);
  } else if (bv.region) {
    // [west, south, east, north, minHeight, maxHeight] in radians + meters
    const r = bv.region;
    const corners = [
      [r[0], r[1], r[4]],
      [r[0], r[1], r[5]],
      [r[0], r[3], r[4]],
      [r[0], r[3], r[5]],
      [r[2], r[1], r[4]],
      [r[2], r[1], r[5]],
      [r[2], r[3], r[4]],
      [r[2], r[3], r[5]]
    ];
    for (const c of corners) {
      // we'll handle region bounds lazily via lon/lat -> ECEF; for now
      // we use a wide box because regions are rare in oblique photogrammetry.
      const RAD2DEG = 180 / Math.PI;
      // Approximate: 1 deg lat = ~111 km. For testing, fall back to large box.
      const cx = c[0] * RAD2DEG * 111000;
      const cy = c[1] * RAD2DEG * 111000;
      box.min.x = Math.min(box.min.x, cx);
      box.min.y = Math.min(box.min.y, cy);
      box.max.x = Math.max(box.max.x, cx);
      box.max.y = Math.max(box.max.y, cy);
    }
    box.min.z = r[4];
    box.max.z = r[5];
  }
  return box;
}

function boxIntersects(a, b) {
  if (a.isEmpty() || b.isEmpty()) return false;
  return !(
    a.max.x < b.min.x || a.min.x > b.max.x ||
    a.max.y < b.min.y || a.min.y > b.max.y ||
    a.max.z < b.min.z || a.min.z > b.max.z
  );
}

// Walk the tileset, return all leaf tiles (or those with .content) whose
// bounding box intersects the given query box. `baseUrl` is the URL of
// the file that contains this subtree, used to resolve relative content
// URLs. This is async because inner tilesets may need to be fetched
// (their URL ends in .json).
//
// We stop recursing when:
//  - maxDepth is reached, OR
//  - the tile's geometricError is already small enough for the query
//    box size (i.e. further subdivision would not improve resolution
//    relative to the area we care about).
async function collectLeafTiles(root, queryBox, out, baseUrl, fetchJson, depth = 0, maxDepth = 25) {
  if (!root || depth > maxDepth) return;
  const box = makeBoundingBox3(root.boundingVolume);
  if (box.isEmpty()) return;
  if (!boxIntersects(box, queryBox)) return;

  // Stop condition: geometric error is small enough relative to the
  // query box. We want a few pixels of mesh per meter of scene, so
  // stopping at ~0.25 m geometric error is reasonable for the
  // 30m x 30m default bbox.
  const minAcceptableError = 0.5;
  if (root.geometricError !== undefined && root.geometricError < minAcceptableError && depth > 2) {
    // Use this tile's content (b3dm) as the leaf
    if (root.content && root.content.url) {
      const ref = root.content.url;
      if (!/\.json($|\?)/i.test(ref)) {
        out.push({ tile: root, baseUrl });
        return;
      }
    }
  }

  // If this node has a content URL pointing to a sub-tileset JSON, the
  // node is essentially a reference to that sub-tileset. We always
  // recurse into the sub-tileset when its bounding box intersects our
  // query box.
  if (root.content && root.content.url) {
    const ref = root.content.url;
    const isJson = /\.json($|\?)/i.test(ref);
    if (isJson) {
      let url = ref;
      if (!url.startsWith('http') && !url.startsWith('blob')) {
        url = baseUrl + url;
      }
      try {
        const sub = await fetchJson(url);
        const idx = url.lastIndexOf('/');
        const subBase = idx >= 0 ? url.substring(0, idx + 1) : url + '/';
        const subRoot = sub.root || sub;
        await collectLeafTiles(subRoot, queryBox, out, subBase, fetchJson, depth + 1, maxDepth);
      } catch (err) {
        logger.warn(`子 tileset 加载失败 ${url}: ${err.message}`);
      }
      return;
    } else {
      // Binary content (b3dm) at this node with no children - record as leaf
      if (!root.children || root.children.length === 0) {
        out.push({ tile: root, baseUrl });
        return;
      }
    }
  }

  // Descend into children.
  if (root.children && root.children.length > 0) {
    for (const child of root.children) {
      await collectLeafTiles(child, queryBox, out, baseUrl, fetchJson, depth + 1, maxDepth);
    }
  }
}

export class Tileset {
  constructor() {
    this.url = null;
    this.root = null;
    this.group = new THREE.Group();
    this.group.name = 'tileset';
    this.tiles = new Map();          // url -> { tileJson, gltf, lastUsed }
    this.lruOrder = [];              // urls in LRU order
    this.maxCached = 12;
  }

  async load(tilesetUrl) {
    this.url = tilesetUrl;
    const r = await fetchWithRetry(tilesetUrl);
    this.root = await r.json();
    logger.info(`已加载 tileset.json，根节点 boundingSphere 半径 = ${this.root.root.boundingVolume.sphere[3].toFixed(1)} m`);
    return this;
  }

  getBoundingBox() {
    return makeBoundingBox3(this.root.root.boundingVolume);
  }

  // Returns an array of Three.js Object3D (meshes) whose world boxes
  // intersect queryBox (in ECEF). Loads tiles lazily, evicts old ones.
  async getMeshesInBox(queryBox) {
    const leaves = [];
    const baseUrl = this.url.substring(0, this.url.lastIndexOf('/') + 1);
    const fetchJson = async (u) => {
      const r = await fetchWithRetry(u);
      return r.json();
    };
    await collectLeafTiles(this.root.root, queryBox, leaves, baseUrl, fetchJson);
    if (leaves.length === 0) {
      logger.warn('指定包围盒内未找到任何 3D Tiles 瓦片');
      return [];
    }
    logger.info(`包围盒内候选瓦片 ${leaves.length} 个，开始加载...`);

    const loaded = [];
    let i = 0;
    for (const { tile, baseUrl: tileBase } of leaves) {
      i++;
      let url = tile.content.url;
      if (!url.startsWith('http') && !url.startsWith('blob')) {
        url = tileBase + url;
      }
      try {
        const t0 = performance.now();
        const gltf = await loadGlb(url);
        const dt = performance.now() - t0;
        logger.info(`已加载 ${url.split('/').pop()} (${dt.toFixed(0)} ms, 网格 ${gltf.scene.children.length})`);
        if (this.tiles.has(url)) {
          this.tiles.get(url).lastUsed = performance.now();
        } else {
          this.tiles.set(url, { tile, gltf, lastUsed: performance.now() });
        }
        // Evict LRU if needed
        while (this.tiles.size > this.maxCached) {
          let oldestUrl = null;
          let oldestTime = Infinity;
          for (const [u, e] of this.tiles) {
            if (e.lastUsed < oldestTime) {
              oldestTime = e.lastUsed;
              oldestUrl = u;
            }
          }
          if (oldestUrl) {
            this.tiles.delete(oldestUrl);
            logger.info(`淘汰缓存瓦片: ${oldestUrl.split('/').pop()}`);
          }
        }
        // Add meshes to group if not already present
        if (!this.group.children.find(c => c.userData.url === url)) {
          const obj = gltf.scene;
          obj.userData.url = url;
          this.group.add(obj);
        }
        for (const child of gltf.scene.children) {
          loaded.push({ mesh: child, tile, url });
        }
        if (i % 5 === 0 || i === leaves.length) {
          logger.info(`瓦片加载进度: ${i}/${leaves.length}`);
        }
      } catch (err) {
        logger.warn(`瓦片加载失败 ${url.split('/').pop()}: ${err.message}`);
      }
    }
    return loaded;
  }

  // Return a single Three.js BufferGeometry merging all meshes in the
  // given list. Vertices are kept in ECEF (world) coordinates.
  buildMergedGeometry(meshes) {
    const positions = [];
    const normals = [];
    const colors = [];
    const indices = [];
    let vertexOffset = 0;

    for (const { mesh } of meshes) {
      mesh.updateWorldMatrix(true, false);
      const m = mesh.matrixWorld;
      const n = new THREE.Matrix3().getNormalMatrix(m);

      mesh.traverse((o) => {
        if (!o.isMesh || !o.geometry) return;
        const geom = o.geometry;
        const posAttr = geom.attributes.position;
        if (!posAttr) return;
        const normAttr = geom.attributes.normal;
        const colAttr = geom.attributes.color;
        const idx = geom.index;
        const vcount = posAttr.count;
        for (let i = 0; i < vcount; i++) {
          const p = new THREE.Vector3().fromBufferAttribute(posAttr, i).applyMatrix4(m);
          positions.push(p.x, p.y, p.z);
          if (normAttr) {
            const nv = new THREE.Vector3().fromBufferAttribute(normAttr, i).applyMatrix3(n).normalize();
            normals.push(nv.x, nv.y, nv.z);
          } else {
            normals.push(0, 1, 0);
          }
          if (colAttr) {
            const c = new THREE.Color().fromBufferAttribute(colAttr, i);
            // Vertex color is often in linear space; keep sRGB-encoded
            colors.push(c.r, c.g, c.b);
          } else {
            // Use a default mid-gray if no per-vertex color
            colors.push(0.7, 0.7, 0.7);
          }
        }
        if (idx) {
          for (let i = 0; i < idx.count; i++) {
            indices.push(idx.getX(i) + vertexOffset);
          }
        } else {
          for (let i = 0; i < vcount; i += 3) {
            indices.push(vertexOffset + i, vertexOffset + i + 1, vertexOffset + i + 2);
          }
        }
        vertexOffset += vcount;
      });
    }

    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    merged.setAttribute('normal',   new THREE.Float32BufferAttribute(normals, 3));
    merged.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3));
    merged.setIndex(indices);
    return merged;
  }
}
