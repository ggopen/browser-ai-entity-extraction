// Direct test of the glTF 1.0 binary parser using a mock b3dm file.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as fs from 'fs';
import { Buffer } from 'buffer';

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

function parseGlbBinary(glbBuf) {
  const dv = new DataView(glbBuf);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546C67) throw new Error('Not a GLB');
  const length = dv.getUint32(8, true);
  let offset = 12;
  let json = null;
  let binChunks = [];
  while (offset < length) {
    const chunkLen = dv.getUint32(offset, true); offset += 4;
    const chunkType = dv.getUint32(offset, true); offset += 4;
    if (chunkType === 0x4E4F534A) {
      const raw = Buffer.from(new Uint8Array(glbBuf, offset, chunkLen));
      let end = raw.length;
      while (end > 0 && raw[end - 1] <= 0x20) end--;
      json = JSON.parse(raw.subarray(0, end).toString('utf-8'));
    } else if (chunkType === 0x004E4942) {
      binChunks.push({ offset, byteLength: chunkLen });
    }
    offset += chunkLen + ((4 - (chunkLen % 4)) % 4);
  }
  console.log('Parsed JSON:', JSON.stringify(json, null, 2).slice(0, 500));
  const bin = binChunks[0];
  const geometries = [];
  for (const mesh of json.meshes || []) {
    for (const prim of mesh.primitives || []) {
      const g = new THREE.BufferGeometry();
      const attrs = prim.attributes || {};
      for (const [name, accIdx] of Object.entries(attrs)) {
        const acc = json.accessors[accIdx];
        const TypedArray = COMPONENT_TYPE[acc.componentType];
        console.log('  Attribute', name, 'componentType', acc.componentType, 'TypedArray', TypedArray);
        const arr = new TypedArray(
          glbBuf,
          bin.offset + (json.bufferViews[acc.bufferView].byteOffset || 0) + (acc.byteOffset || 0),
          acc.count * TYPE_SIZE[acc.type]
        );
        console.log('    Array length:', arr.length);
        g.setAttribute(name, new THREE.BufferAttribute(arr, TYPE_SIZE[acc.type]));
        console.log('    attr count:', g.attributes[name].count);
      }
      if (prim.indices !== undefined) {
        const acc = json.accessors[prim.indices];
        const arr = new COMPONENT_TYPE[acc.componentType](
          glbBuf,
          bin.offset + (json.bufferViews[acc.bufferView].byteOffset || 0) + (acc.byteOffset || 0),
          acc.count
        );
        g.setIndex(new THREE.BufferAttribute(arr, 1));
      }
      geometries.push(g);
    }
  }
  return geometries;
}

const b3dm = fs.readFileSync('./mock-tileset/tile.b3dm');
const dv = new DataView(b3dm.buffer, b3dm.byteOffset, b3dm.length);
const b3dmMagic = dv.getUint32(0, true);
const b3dmVersion = dv.getUint32(4, true);
const b3dmByteLength = dv.getUint32(8, true);
const ftJsonLen = dv.getUint32(12, true);
const ftBinLen = dv.getUint32(16, true);
const btJsonLen = dv.getUint32(20, true);
const btBinLen = dv.getUint32(24, true);
const glbOffset = 28 + ftJsonLen + ftBinLen + btJsonLen + btBinLen;
console.log('b3dm magic:', b3dmMagic.toString(16), 'version:', b3dmVersion, 'length:', b3dmByteLength);
console.log('glbOffset:', glbOffset);
const glb = b3dm.buffer.slice(b3dm.byteOffset + glbOffset, b3dm.byteOffset + b3dm.byteLength);

const geoms = parseGlbBinary(glb);
console.log('Geometries:', geoms.length);
for (const g of geoms) {
  console.log('  attrs keys:', Object.keys(g.attributes || {}));
  console.log('  Vertices:', g.attributes.position ? g.attributes.position.count : 'no position');
}

// Test that GLTFLoader rejects it
const loader = new GLTFLoader();
try {
  loader.parse(glb, '', (g) => console.log('GLTFLoader success'), (e) => console.log('GLTFLoader error callback:', e.message));
} catch (e) {
  console.log('GLTFLoader threw:', e.message);
}
