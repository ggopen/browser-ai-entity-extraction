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
    const jsonBytes = new Uint8Array(arrayBuffer, 28, ftJsonLen);
    header.featureTable = JSON.parse(new TextDecoder('utf-8').decode(jsonBytes));
  }
  if (btJsonLen > 0) {
    const jsonBytes = new Uint8Array(arrayBuffer, 28 + ftJsonLen + ftBinLen, btJsonLen);
    header.batchTable = JSON.parse(new TextDecoder('utf-8').decode(jsonBytes));
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
  // Walk chunks
  let offset = 12;
  let json = null;
  let binChunks = []; // { offset, byteLength }
  while (offset < length) {
    const chunkLen = dv.getUint32(offset, true); offset += 4;
    const chunkType = dv.getUint32(offset, true); offset += 4;
    const data = new Uint8Array(glbBuf, offset, chunkLen);
    if (chunkType === 0x4E4F534A) { // 'JSON'
      json = JSON.parse(new TextDecoder('utf-8').decode(data));
    } else if (chunkType === 0x004E4942) { // 'BIN\0'
      binChunks.push({ offset, byteLength: chunkLen });
    }
    // Pad to 4 bytes
    offset += chunkLen + ((4 - (chunkLen % 4)) % 4);
  }
  if (!json) throw new Error('GLB has no JSON chunk');

  const group = new THREE.Group();
  const bin = binChunks[0];
  if (!bin) return group;

  // Helper: read accessor as a typed array view into the BIN data
  function readAccessor(acc) {
    const bv = json.bufferViews[acc.bufferView];
    const compT = acc.componentType;
    const TypedArray = COMPONENT_TYPE[compT];
    const elemBytes = COMPONENT_BYTES[compT];
    const typeCount = TYPE_SIZE[acc.type];
    const total = acc.count * typeCount;
    const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
    return new TypedArray(glbBuf, bin.offset + byteOffset, total);
  }

  // For each mesh primitive, build a Three.js BufferGeometry
  for (const mesh of json.meshes || []) {
    for (const prim of mesh.primitives || []) {
      try {
        const g = new THREE.BufferGeometry();
        // Attributes
        const attrs = prim.attributes || {};
        for (const [name, accIdx] of Object.entries(attrs)) {
          const acc = json.accessors[accIdx];
          const arr = readAccessor(acc);
          const itemSize = TYPE_SIZE[acc.type];
          // Three.js naming: POSITION, NORMAL, COLOR_0, TEXCOORD_0...
          g.setAttribute(name, new THREE.BufferAttribute(arr, itemSize));
        }
        // Indices
        if (prim.indices !== undefined) {
          const acc = json.accessors[prim.indices];
          const arr = readAccessor(acc);
          const itemSize = TYPE_SIZE[acc.type];
          g.setIndex(new THREE.BufferAttribute(arr, itemSize));
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

function loadGlb(url) {
  if (gltfCache.has(url)) return gltfCache.get(url);
  const promise = fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      return r.arrayBuffer();
    })
    .then((buf) => {
      const header = parseB3dmHeader(buf);
      const glbBuf = extractGlb(buf, header.glbOffset);
      const rtc = buildRtcTransform(header.featureTable);

      // Try GLTFLoader first (glTF 2.0 binary); fall back to our
      // custom glTF 1.0 binary parser on failure.
      return new Promise((resolve, reject) => {
        gltfLoader.parse(
          glbBuf,
          '',
          (gltf) => {
            gltf.scene.traverse((o) => {
              if (o.isMesh && o.geometry) {
                o.geometry.applyMatrix4(rtc);
              }
            });
            resolve(gltf);
          },
          () => {
            // glTF 1.0 binary path
            try {
              const group = parseGlbBinary(glbBuf);
              group.traverse((o) => {
                if (o.isMesh && o.geometry) {
                  o.geometry.applyMatrix4(rtc);
                }
              });
              resolve({ scene: group });
            } catch (e) {
              reject(e);
            }
          }
        );
      });
    });
  gltfCache.set(url, promise);
  return promise;
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
    const r = await fetch(tilesetUrl);
    if (!r.ok) throw new Error(`Failed to load tileset.json: HTTP ${r.status}`);
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
      const r = await fetch(u);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
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
        // Touch LRU
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
