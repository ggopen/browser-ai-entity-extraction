// Multi-view off-screen renderer.
// For each view, produces:
//   - an RGB image (RGBA8)
//   - a vertex-id image (RGBA8, encoding a 32-bit triangle id per pixel)

import * as THREE from 'three';
import { logger } from './logger.js';

const MAX_VERTEX_ID = 0xFFFFFFFF;

function makeVertexIdMaterial() {
  // WebGL2 GLSL ES 3.00. Uses gl_VertexID and flat interpolation so the
  // rasterized fragment carries the integer id of the last vertex of
  // the triangle (one id per triangle).
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: /* glsl */`
      flat out uint vId;
      void main() {
        vId = uint(gl_VertexID);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      precision highp float;
      flat in uint vId;
      out vec4 fragColor;
      void main() {
        uint id = vId;
        fragColor = vec4(
          float(id & 255u) / 255.0,
          float((id >> 8u) & 255u) / 255.0,
          float((id >> 16u) & 255u) / 255.0,
          float((id >> 24u) & 255u) / 255.0
        );
      }
    `,
    side: THREE.FrontSide,
    depthTest: true,
    depthWrite: true,
    transparent: false
  });
}

function makeRgbMaterial(unlit) {
  // Use the geometry's vertex color or fall back to a flat color.
  return new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.FrontSide,
    depthTest: true,
    depthWrite: true
  });
}

function encodeId(id) {
  // Encode a JS integer as [r, g, b, a] in 0..255.
  return [
    (id)         & 0xff,
    (id >>> 8)   & 0xff,
    (id >>> 16)  & 0xff,
    (id >>> 24)  & 0xff
  ];
}

export function decodeId(r, g, b, a) {
  return (r | (g << 8) | (b << 16) | (a << 24)) >>> 0;
}

function makeViewCameras(distance, centerEcef) {
  // 1 top-down + 4 oblique (0, 90, 180, 270) at 45 deg pitch.
  // Returns an array of { name, camera, lookAt }
  const camList = [
    { name: 'top',   pitch: -89.5, yaw: 0 },
    { name: 'n',     pitch: -45,   yaw: 0 },
    { name: 'e',     pitch: -45,   yaw: 90 },
    { name: 's',     pitch: -45,   yaw: 180 },
    { name: 'w',     pitch: -45,   yaw: 270 }
  ];
  const cameras = [];
  for (const c of camList) {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, distance * 4);
    const pitchRad = (c.pitch * Math.PI) / 180;
    const yawRad = (c.yaw * Math.PI) / 180;
    const offsetX = distance * Math.cos(pitchRad) * Math.cos(yawRad);
    const offsetY = distance * Math.cos(pitchRad) * Math.sin(yawRad);
    const offsetZ = distance * Math.sin(pitchRad);
    cam.position.set(
      centerEcef.x + offsetX,
      centerEcef.y + offsetY,
      centerEcef.z + offsetZ
    );
    cam.up.set(0, 0, 1);
    cam.lookAt(centerEcef.x, centerEcef.y, centerEcef.z);
    cam.updateMatrixWorld();
    cameras.push({ name: c.name, camera: cam });
  }
  return cameras;
}

export class MultiViewRenderer {
  constructor(renderer) {
    this.renderer = renderer;
  }

  // Render the merged local mesh from multiple viewpoints.
  // mesh: a Three.js Object3D positioned in world space.
  // returns Array<{ name, rgb: Uint8ClampedArray(width*height*4), idMap: Uint8ClampedArray(w*h*4), width, height }>
  render(mesh, centerEcef, bboxDiag, options = {}) {
    const viewCount = options.viewCount || 5;
    const res = options.resolution || 512;
    const distance = bboxDiag * 1.2;

    // Build view cameras - pick first viewCount
    const allCams = makeViewCameras(distance, centerEcef);
    const cams = allCams.slice(0, Math.min(viewCount, allCams.length));

    // Choose a subset if requested
    let chosen = cams;
    if (viewCount === 3) chosen = [cams[0], cams[1], cams[3]]; // top, n, s
    if (viewCount === 6) {
      // top + 5 oblique (0/72/144/216/288)
      const extra = [];
      for (let i = 0; i < 5; i++) {
        const yaw = (i * 72);
        const cam = new THREE.PerspectiveCamera(60, 1, 0.1, distance * 4);
        const pitchRad = (-45 * Math.PI) / 180;
        const yawRad = (yaw * Math.PI) / 180;
        cam.position.set(
          centerEcef.x + distance * Math.cos(pitchRad) * Math.cos(yawRad),
          centerEcef.y + distance * Math.cos(pitchRad) * Math.sin(yawRad),
          centerEcef.z + distance * Math.sin(pitchRad)
        );
        cam.up.set(0, 0, 1);
        cam.lookAt(centerEcef.x, centerEcef.y, centerEcef.z);
        cam.updateMatrixWorld();
        extra.push({ name: `o${i}`, camera: cam });
      }
      chosen = [cams[0], ...extra];
    }

    // Build a clone of the mesh with the right materials for id rendering
    const idMaterial = makeVertexIdMaterial();
    const idScene = new THREE.Scene();
    const rgbScene = new THREE.Scene();

    // The mesh's geometry is shared. For id rendering, we add the same
    // mesh to the id scene with the id material. For RGB rendering we
    // need per-vertex colors which the original mesh already has (vertex
    // colors from the b3dm glTF). If vertex colors are missing, we use
    // a flat color.
    const idObj = mesh.clone();
    idObj.traverse((o) => {
      if (o.isMesh) {
        o.material = idMaterial;
        o.frustumCulled = false;
      }
    });
    idScene.add(idObj);

    const rgbObj = mesh.clone();
    rgbObj.traverse((o) => {
      if (o.isMesh) {
        if (!o.geometry.attributes.color) {
          // If no per-vertex color, use a base color
          o.material = new THREE.MeshBasicMaterial({ color: 0xb0b0b0, side: THREE.FrontSide });
        } else {
          o.material = makeRgbMaterial();
        }
        o.frustumCulled = false;
      }
    });
    rgbScene.add(rgbObj);

    // Render targets
    const rgbTarget = new THREE.WebGLRenderTarget(res, res, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true
    });
    const idTarget = new THREE.WebGLRenderTarget(res, res, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true
    });

    const prevTarget = this.renderer.getRenderTarget();
    const prevAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = true;

    const out = [];
    const w = res, h = res;
    for (const c of chosen) {
      c.camera.aspect = 1;
      c.camera.updateProjectionMatrix();

      // RGB pass
      this.renderer.setRenderTarget(rgbTarget);
      this.renderer.clear(true, true, true);
      this.renderer.render(rgbScene, c.camera);
      const rgbBuf = new Uint8Array(w * h * 4);
      this.renderer.readRenderTargetPixels(rgbTarget, 0, 0, w, h, rgbBuf);

      // ID pass
      this.renderer.setRenderTarget(idTarget);
      this.renderer.clear(true, true, true);
      this.renderer.render(idScene, c.camera);
      const idBuf = new Uint8Array(w * h * 4);
      this.renderer.readRenderTargetPixels(idTarget, 0, 0, w, h, idBuf);

      out.push({
        name: c.name,
        camera: c.camera,
        rgb: new Uint8ClampedArray(rgbBuf.buffer),
        idMap: new Uint8ClampedArray(idBuf.buffer),
        width: w,
        height: h
      });
    }

    this.renderer.setRenderTarget(prevTarget);
    this.renderer.autoClear = prevAutoClear;

    // Clean up
    rgbTarget.dispose();
    idTarget.dispose();
    idMaterial.dispose();
    rgbObj.traverse((o) => {
      if (o.isMesh) o.material.dispose();
    });

    return out;
  }
}
