// Simulate the browser's GLB parsing
import * as fs from 'fs';
import { Buffer } from 'buffer';

const b3dm = fs.readFileSync('./mock-tileset/tile.b3dm');
const dv = new DataView(b3dm.buffer, b3dm.byteOffset, b3dm.length);
const ftj_len = dv.getUint32(12, true);
const ftb_len = dv.getUint32(16, true);
const btj_len = dv.getUint32(20, true);
const btb_len = dv.getUint32(24, true);
const glb_offset = 28 + ftj_len + ftb_len + btj_len + btb_len;
const glb = b3dm.buffer.slice(b3dm.byteOffset + glb_offset, b3dm.byteOffset + b3dm.byteLength);
console.log('glb offset:', glb_offset, 'glb length:', glb.byteLength);

// Now simulate GLTFBinaryExtension (three.js r160)
const BINARY_EXTENSION_HEADER_LENGTH = 12;
const headerView = new DataView(glb, 0, BINARY_EXTENSION_HEADER_LENGTH);
const textDecoder = new TextDecoder();
const header = {
  magic: textDecoder.decode(new Uint8Array(glb.slice(0, 4))),
  version: headerView.getUint32(4, true),
  length: headerView.getUint32(8, true)
};
console.log('GLB header:', header);

if (header.version < 2.0) {
  console.log('THREE.GLTFLoader: Legacy binary file detected.');
}

// Now simulate our custom parser
const dv2 = new DataView(glb);
const length = dv2.getUint32(8, true);
let offset = 12;
let json = null;
let binChunks = [];
while (offset < length) {
  const chunkLen = dv2.getUint32(offset, true); offset += 4;
  const chunkType = dv2.getUint32(offset, true); offset += 4;
  console.log('Chunk at', offset, 'len', chunkLen, 'type', chunkType.toString(16));
  if (chunkType === 0x4E4F534A) {
    const raw = new Uint8Array(glb, offset, chunkLen);
    let end = chunkLen;
    while (end > 0 && raw[end - 1] <= 0x20) end--;
    const text = new TextDecoder('utf-8').decode(raw.subarray(0, end));
    console.log('JSON text (first 50):', text.slice(0, 50), 'len:', text.length);
    try {
      json = JSON.parse(text);
      console.log('JSON parsed OK');
    } catch (e) {
      console.log('JSON parse error:', e.message);
      console.log('Text around 18:', JSON.stringify(text.slice(15, 25)));
    }
  } else if (chunkType === 0x004E4942) {
    binChunks.push({ offset, byteLength: chunkLen });
  }
  offset += chunkLen + ((4 - (chunkLen % 4)) % 4);
}
