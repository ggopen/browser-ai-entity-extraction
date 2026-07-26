// Export utilities: GLB (binary glTF) and GeoJSON (boundary polygon).

import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { logger } from './logger.js';
import { ecefToLonLatAlt, buildLocalFrame } from './coord.js';

export function exportGlb(geometry, fileName = 'entity.glb') {
  // The geometry is already in ECEF; export as a binary glTF.
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const group = new THREE.Group();
  group.add(mesh);
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(
      group,
      (result) => {
        if (result instanceof ArrayBuffer) {
          const blob = new Blob([result], { type: 'model/gltf-binary' });
          triggerDownload(blob, fileName);
          resolve();
        } else {
          // Fallback: write as JSON glTF
          const json = JSON.stringify(result, null, 2);
          const blob = new Blob([json], { type: 'model/gltf+json' });
          triggerDownload(blob, fileName.replace(/\.glb$/, '.gltf'));
          resolve();
        }
      },
      (err) => {
        logger.err(`GLB 导出失败: ${err.message || err}`);
        reject(err);
      },
      { binary: true }
    );
  });
}

export function exportGeoJSON(geometry, fileName = 'entity.geojson') {
  // Compute the horizontal (XY in local frame) boundary of the entity.
  // Use the average ECEF point as the local frame origin.
  const pos = geometry.attributes.position;
  if (!pos || pos.count === 0) {
    throw new Error('几何体为空');
  }
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < pos.count; i++) {
    cx += pos.getX(i);
    cy += pos.getY(i);
    cz += pos.getZ(i);
  }
  cx /= pos.count;
  cy /= pos.count;
  cz /= pos.count;
  const center = ecefToLonLatAlt(cx, cy, cz);
  const frame = buildLocalFrame(center.lon, center.lat, center.alt);

  // Project each vertex to local frame, then to 2D (east, north).
  const local2d = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const lp = frame.toLocal(x, y, z);
    local2d.push([lp.e, lp.n]);
  }

  // Compute 2D alpha-shape (here simplified to convex hull)
  const hull = convexHull(local2d);
  if (hull.length < 3) {
    throw new Error('实体顶点数不足以构成多边形');
  }
  // Convert 2D local (east, north) back to lon/lat
  const ring = hull.map(([e, n]) => {
    const ecef = frame.toECEF(e, n, 0);
    const lla = ecefToLonLatAlt(ecef.x, ecef.y, ecef.z);
    return [lla.lon, lla.lat];
  });
  // Close the ring
  ring.push(ring[0]);

  const minZ = minArr(pos, 'z');
  const maxZ = maxArr(pos, 'z');
  const height = maxZ - minZ;
  const area = polygonArea(ring);

  const feature = {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [ring]
    },
    properties: {
      height_m: height,
      area_m2: area,
      center_lon: center.lon,
      center_lat: center.lat,
      elevation_m: center.alt
    }
  };
  const fc = { type: 'FeatureCollection', features: [feature] };
  const json = JSON.stringify(fc, null, 2);
  const blob = new Blob([json], { type: 'application/geo+json' });
  triggerDownload(blob, fileName);
  return feature.properties;
}

function minArr(attr, axis) {
  let m = Infinity;
  for (let i = 0; i < attr.count; i++) m = Math.min(m, attr['get' + axis.toUpperCase()](i));
  return m;
}
function maxArr(attr, axis) {
  let m = -Infinity;
  for (let i = 0; i < attr.count; i++) m = Math.max(m, attr['get' + axis.toUpperCase()](i));
  return m;
}

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
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
function polygonArea(ring) {
  let s = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const a = ring[i];
    const b = ring[i + 1];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(s) / 2;
}

function triggerDownload(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 200);
}
