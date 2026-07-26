// 2D -> 3D voting and entity extraction.
// Given multi-view instance labels, project each triangle centroid to
// each view and tally votes. The triangle closest to the click point
// is the "pivot" and all triangles with the same majority-vote instance
// form the extracted entity.

import * as THREE from 'three';
import { logger } from './logger.js';
import { decodeId } from './multi-view-renderer.js';

function projectCentroid(view, viewRes, ecef) {
  // Returns { x: pixelX, y: pixelY, z: depth } or null if outside view
  const ndc = ecef.clone().project(view.camera);
  if (ndc.z < -1 || ndc.z > 1) return null;
  // ndc in [-1, 1]
  const x = (ndc.x * 0.5 + 0.5) * viewRes;
  const y = (1 - (ndc.y * 0.5 + 0.5)) * viewRes;
  return { x, y, z: ndc.z };
}

function pixelLabel(view, x, y) {
  if (x < 0 || y < 0 || x >= view.width || y >= view.height) return 0;
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const i = (yi * view.width + xi) * 4;
  return view.labels[i];
}

// For each triangle in the merged geometry, compute its centroid in
// world (ECEF) coords, then sample the label at that pixel in each
// of the provided views. Returns a votes matrix: votes[ti] = Int32Array
// of per-view labels (0 if not visible).
export function voteTriangles(mergedGeometry, views) {
  const pos = mergedGeometry.attributes.position;
  const index = mergedGeometry.index;
  const triCount = index ? index.count / 3 : pos.count / 3;
  const triCentroids = new Float32Array(triCount * 3);
  const triLabels = new Int32Array(triCount);
  const v = new THREE.Vector3();
  const c = new THREE.Vector3();

  for (let ti = 0; ti < triCount; ti++) {
    let i0, i1, i2;
    if (index) {
      i0 = index.getX(ti * 3);
      i1 = index.getX(ti * 3 + 1);
      i2 = index.getX(ti * 3 + 2);
    } else {
      i0 = ti * 3;
      i1 = ti * 3 + 1;
      i2 = ti * 3 + 2;
    }
    v.fromBufferAttribute(pos, i0);
    c.copy(v);
    v.fromBufferAttribute(pos, i1); c.add(v);
    v.fromBufferAttribute(pos, i2); c.add(v);
    c.multiplyScalar(1 / 3);
    triCentroids[ti * 3]     = c.x;
    triCentroids[ti * 3 + 1] = c.y;
    triCentroids[ti * 3 + 2] = c.z;

    // Tally votes across views (each view's contribution = 1)
    const tally = new Map();
    for (const view of views) {
      const ecef = new THREE.Vector3(c.x, c.y, c.z);
      const pix = projectCentroid(view, view.width, ecef);
      if (!pix) continue;
      const label = pixelLabel(view, pix.x, pix.y);
      if (label === 0) continue;
      tally.set(label, (tally.get(label) || 0) + 1);
    }
    // Pick majority label
    let best = 0;
    let bestCount = 0;
    for (const [lab, cnt] of tally) {
      if (cnt > bestCount) { best = lab; bestCount = cnt; }
    }
    triLabels[ti] = best;
  }
  return { triCount, triCentroids, triLabels };
}

// Find the triangle closest to a given click point (in ECEF).
// Returns the triangle index.
export function findClickTriangle(mergedGeometry, triCentroids, clickEcef) {
  const pos = mergedGeometry.attributes.position;
  const index = mergedGeometry.index;
  const triCount = index ? index.count / 3 : pos.count / 3;
  let bestTi = -1;
  let bestDist = Infinity;
  const c = new THREE.Vector3();
  for (let ti = 0; ti < triCount; ti++) {
    const cx = triCentroids[ti * 3];
    const cy = triCentroids[ti * 3 + 1];
    const cz = triCentroids[ti * 3 + 2];
    const dx = cx - clickEcef.x;
    const dy = cy - clickEcef.y;
    const dz = cz - clickEcef.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestDist) {
      bestDist = d2;
      bestTi = ti;
    }
  }
  return bestTi;
}

// Extract all triangle indices that share the same label as the given
// reference triangle. Optionally also include triangles within a
// distance threshold of the reference (for fallback / boundary cases).
export function extractEntityTriangles(triLabels, refLabel, refTi, triCentroids, maxRadius = 5) {
  const indices = [];
  const triCount = triLabels.length;
  if (refLabel > 0) {
    for (let ti = 0; ti < triCount; ti++) {
      if (triLabels[ti] === refLabel) indices.push(ti);
    }
    return indices;
  }
  // Fallback: include nearby triangles
  const refCx = triCentroids[refTi * 3];
  const refCy = triCentroids[refTi * 3 + 1];
  const refCz = triCentroids[refTi * 3 + 2];
  const r2 = maxRadius * maxRadius;
  for (let ti = 0; ti < triCount; ti++) {
    if (triLabels[ti] === 0) continue;
    const dx = triCentroids[ti * 3]     - refCx;
    const dy = triCentroids[ti * 3 + 1] - refCy;
    const dz = triCentroids[ti * 3 + 2] - refCz;
    if (dx * dx + dy * dy + dz * dz <= r2) indices.push(ti);
  }
  return indices;
}

// Build a sub-geometry from selected triangle indices.
export function buildEntityGeometry(mergedGeometry, triIndices) {
  const pos = mergedGeometry.attributes.position;
  const norm = mergedGeometry.attributes.normal;
  const col = mergedGeometry.attributes.color;
  const index = mergedGeometry.index;
  const newPos = [];
  const newNorm = [];
  const newCol = [];
  const newIdx = [];
  let off = 0;
  for (const ti of triIndices) {
    let i0, i1, i2;
    if (index) {
      i0 = index.getX(ti * 3);
      i1 = index.getX(ti * 3 + 1);
      i2 = index.getX(ti * 3 + 2);
    } else {
      i0 = ti * 3; i1 = ti * 3 + 1; i2 = ti * 3 + 2;
    }
    for (const idx of [i0, i1, i2]) {
      newPos.push(pos.getX(idx), pos.getY(idx), pos.getZ(idx));
      if (norm) newNorm.push(norm.getX(idx), norm.getY(idx), norm.getZ(idx));
      if (col)  newCol.push(col.getX(idx),  col.getY(idx),  col.getZ(idx));
    }
    newIdx.push(off, off + 1, off + 2);
    off += 3;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(newPos, 3));
  if (newNorm.length) g.setAttribute('normal', new THREE.Float32BufferAttribute(newNorm, 3));
  if (newCol.length)  g.setAttribute('color',  new THREE.Float32BufferAttribute(newCol, 3));
  g.setIndex(newIdx);
  return g;
}

// Compute alpha-shape boundary of an entity's vertices projected to a
// horizontal plane. Returns an array of [x, y] points (in local frame)
// forming the boundary polygon (or [] if too few points).
export function alphaShape2D(localPoints, alpha = 0.5) {
  if (localPoints.length < 4) return localPoints;
  // Simple alpha-shape: Delaunay edges filtered by alpha criterion.
  // For simplicity (and to avoid bundling a Delaunay lib), we use the
  // convex hull approximation via Andrew's monotone chain. For oblique
  // photogrammetry entities (mostly buildings, trees), the convex hull
  // is usually a good-enough approximation of the alpha-shape when the
  // building footprint is approximately convex.
  const pts = localPoints.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const n = pts.length;
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

// Shoelace polygon area (local frame units^2)
export function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(s) / 2;
}
