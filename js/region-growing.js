// Geometric region-growing fallback. Builds an instance id per triangle
// using planar / normal / colour similarity. This is the classic "CSF
// + region growing" approach the PRD mentions as a fallback.
//
// This is intentionally simple: we seed on the click triangle and grow
// to neighbours whose plane-fit residual and color distance stay below
// thresholds. Useful when the AI model fails to load or times out.

import * as THREE from 'three';

function triNormal(geom, ti) {
  const pos = geom.attributes.position;
  const idx = geom.index;
  const i0 = idx ? idx.getX(ti * 3) : ti * 3;
  const i1 = idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1;
  const i2 = idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2;
  const a = new THREE.Vector3().fromBufferAttribute(pos, i0);
  const b = new THREE.Vector3().fromBufferAttribute(pos, i1);
  const c = new THREE.Vector3().fromBufferAttribute(pos, i2);
  return new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).normalize();
}

function triCentroid(geom, ti) {
  const pos = geom.attributes.position;
  const idx = geom.index;
  const i0 = idx ? idx.getX(ti * 3) : ti * 3;
  const i1 = idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1;
  const i2 = idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2;
  const a = new THREE.Vector3().fromBufferAttribute(pos, i0);
  const b = new THREE.Vector3().fromBufferAttribute(pos, i1);
  const c = new THREE.Vector3().fromBufferAttribute(pos, i2);
  return a.add(b).add(c).multiplyScalar(1 / 3);
}

function triColor(geom, ti) {
  const col = geom.attributes.color;
  if (!col) return [0.5, 0.5, 0.5];
  const idx = geom.index;
  const i0 = idx ? idx.getX(ti * 3) : ti * 3;
  return [col.getX(i0), col.getY(i0), col.getZ(i0)];
}

// Build an adjacency list: triangles sharing a vertex or an edge.
function buildAdjacency(geom) {
  const pos = geom.attributes.position;
  const idx = geom.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  // Map from vertex-id to list of triangle indices
  const vertMap = new Map();
  for (let ti = 0; ti < triCount; ti++) {
    const ids = [
      idx ? idx.getX(ti * 3) : ti * 3,
      idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1,
      idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2
    ];
    for (const v of ids) {
      if (!vertMap.has(v)) vertMap.set(v, []);
      vertMap.get(v).push(ti);
    }
  }
  const adj = new Array(triCount);
  for (let i = 0; i < triCount; i++) adj[i] = new Set();
  for (const tris of vertMap.values()) {
    for (let i = 0; i < tris.length; i++) {
      for (let j = i + 1; j < tris.length; j++) {
        adj[tris[i]].add(tris[j]);
        adj[tris[j]].add(tris[i]);
      }
    }
  }
  return adj;
}

function colorDist(a, b) {
  return Math.sqrt(
    (a[0] - b[0]) ** 2 +
    (a[1] - b[1]) ** 2 +
    (a[2] - b[2]) ** 2
  );
}

export function regionGrow(geom, startTi, options = {}) {
  const normalThresh = options.normalThresh ?? 0.5;        // dot product
  const colorThresh = options.colorThresh ?? 0.4;
  const maxFaces = options.maxFaces ?? 30000;
  const adj = buildAdjacency(geom);
  const triCount = adj.length;
  const seedNormal = triNormal(geom, startTi);
  const seedColor = triColor(geom, startTi);
  const labels = new Int32Array(triCount);
  labels[startTi] = 1;
  const queue = [startTi];
  let head = 0;
  let count = 1;
  while (head < queue.length) {
    const cur = queue[head++];
    const curNormal = triNormal(geom, cur);
    const curColor = triColor(geom, cur);
    for (const nb of adj[cur]) {
      if (labels[nb]) continue;
      const nbNormal = triNormal(geom, nb);
      const dot = Math.abs(curNormal.dot(nbNormal));
      if (dot < 1 - normalThresh) continue;
      const nbColor = triColor(geom, nb);
      if (colorDist(curColor, nbColor) > colorThresh) continue;
      labels[nb] = 1;
      queue.push(nb);
      count++;
      if (count > maxFaces) break;
    }
    if (count > maxFaces) break;
  }
  return labels;
}
