// Build a mock b3dm 1.0 tile with embedded glTF 1.0 binary GLB.
// Used to test the glTF 1.0 binary parser when the external data
// source is unavailable.
import * as fs from 'fs';
import { Buffer } from 'buffer';

// Generate a simple box geometry data
// 24 vertices (4 per face, 6 faces), 12 triangles
const positions = [
  // +X face
  1, -1, -1,   1,  1, -1,   1,  1,  1,   1, -1,  1,
  // -X face
  -1, -1,  1,  -1,  1,  1,  -1,  1, -1,  -1, -1, -1,
  // +Y face
   1,  1, -1,  -1,  1, -1,  -1,  1,  1,   1,  1,  1,
  // -Y face
   1, -1,  1,  -1, -1,  1,  -1, -1, -1,   1, -1, -1,
  // +Z face
   1,  1,  1,  -1,  1,  1,  -1, -1,  1,   1, -1,  1,
  // -Z face
  -1,  1, -1,   1,  1, -1,   1, -1, -1,  -1, -1, -1
];
const indices = [
  0, 1, 2, 0, 2, 3,
  4, 5, 6, 4, 6, 7,
  8, 9, 10, 8, 10, 11,
  12, 13, 14, 12, 14, 15,
  16, 17, 18, 16, 18, 19,
  20, 21, 22, 20, 22, 23
];

// Build a glTF 1.0 binary GLB.
// Layout:
//   12-byte header (magic, version, length)
//   JSON chunk: 8-byte header (length, 'JSON') + data
//   BIN chunk: 8-byte header (length, 'BIN\0') + data

const posBytes = positions.length * 4;
const idxBytes = indices.length * 4;
const binLength = posBytes + idxBytes;
const jsonObj = {
  asset: { version: '1.0' },
  buffers: [{ byteLength: binLength }],
  bufferViews: [
    { buffer: 0, byteOffset: 0, byteLength: posBytes, target: 34962 },
    { buffer: 0, byteOffset: posBytes, byteLength: idxBytes, target: 34963 }
  ],
  accessors: [
    { bufferView: 0, byteOffset: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min: [-1, -1, -1], max: [1, 1, 1] },
    { bufferView: 1, byteOffset: 0, componentType: 5125, count: indices.length, type: 'SCALAR' }
  ],
  meshes: [{
    primitives: [{
      attributes: { POSITION: 0 },
      indices: 1,
      mode: 4
    }]
  }],
  nodes: [{ meshes: [0] }],
  scenes: [{ nodes: [0] }],
  scene: 0
};
const jsonStr = JSON.stringify(jsonObj);
const jsonBytes = Buffer.from(jsonStr, 'utf-8');
// Pad to 4 bytes
const jsonPadded = Math.ceil(jsonBytes.length / 4) * 4;
const jsonChunkData = Buffer.alloc(jsonPadded);
jsonBytes.copy(jsonChunkData);

// Pad BIN chunk to 4 bytes
const binPadded = Math.ceil(binLength / 4) * 4;
const binChunkData = Buffer.alloc(binPadded);
// Write positions
for (let i = 0; i < positions.length; i++) {
  binChunkData.writeFloatLE(positions[i], i * 4);
}
// Write indices after positions
for (let i = 0; i < indices.length; i++) {
  binChunkData.writeUInt32LE(indices[i], posBytes + i * 4);
}

// Total GLB size: 12 (header) + 8 (json chunk header) + jsonPadded + 8 (bin chunk header) + binPadded
const glbLength = 12 + 8 + jsonPadded + 8 + binPadded;
const glb = Buffer.alloc(glbLength);
let off = 0;
glb.writeUInt32LE(0x46546C67, off); off += 4; // 'glTF'
glb.writeUInt32LE(1, off); off += 4; // version 1
glb.writeUInt32LE(glbLength, off); off += 4;
// JSON chunk
glb.writeUInt32LE(jsonPadded, off); off += 4;
glb.writeUInt32LE(0x4E4F534A, off); off += 4; // 'JSON'
jsonChunkData.copy(glb, off); off += jsonPadded;
// BIN chunk
glb.writeUInt32LE(binPadded, off); off += 4;
glb.writeUInt32LE(0x004E4942, off); off += 4; // 'BIN\0'
binChunkData.copy(glb, off);

console.log('Built GLB:', glbLength, 'bytes, version 1');
console.log('JSON size:', jsonStr.length, '-> padded to', jsonPadded);
console.log('BIN size:', binLength, '-> padded to', binPadded);

// Wrap in b3dm 1.0
// b3dm header (28 bytes): magic, version, byteLength, ftJsonLen, ftBinLen, btJsonLen, btBinLen
const ftJson = JSON.stringify({ BATCH_LENGTH: 0 });
const ftJsonBytes = Buffer.from(ftJson, 'utf-8');
const ftJsonPadded = Math.ceil(ftJsonBytes.length / 4) * 4;
const ftJsonBuf = Buffer.alloc(ftJsonPadded);
ftJsonBytes.copy(ftJsonBuf);

const totalB3dmLen = 28 + ftJsonPadded + glb.length;
const b3dm = Buffer.alloc(totalB3dmLen);
off = 0;
b3dm.writeUInt32LE(0x6D643362, off); off += 4; // b3dm magic
b3dm.writeUInt32LE(1, off); off += 4; // version
b3dm.writeUInt32LE(totalB3dmLen, off); off += 4;
b3dm.writeUInt32LE(ftJsonBuf.length, off); off += 4; // ftJsonLen
b3dm.writeUInt32LE(0, off); off += 4; // ftBinLen
b3dm.writeUInt32LE(0, off); off += 4; // btJsonLen
b3dm.writeUInt32LE(0, off); off += 4; // btBinLen
ftJsonBuf.copy(b3dm, off); off += ftJsonBuf.length;
glb.copy(b3dm, off);

fs.writeFileSync('mock-tileset/tile.b3dm', b3dm);
console.log('Wrote b3dm:', b3dm.length, 'bytes');
