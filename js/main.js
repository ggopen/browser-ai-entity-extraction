// Main application: orchestrates the 3D scene, click-to-extract pipeline,
// UI, and entity management.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { logger } from './logger.js';
import { Tileset } from './tiles-loader.js';
import { MultiViewRenderer } from './multi-view-renderer.js';
import {
  getSegmentationModel, getModelSize, inferBatch
} from './segmentation.js';
import {
  voteTriangles, findClickTriangle, extractEntityTriangles,
  buildEntityGeometry, alphaShape2D, polygonArea
} from './voting.js';
import { regionGrow } from './region-growing.js';
import { exportGlb, exportGeoJSON } from './export.js';
import { lonLatAltToECEF, ecefToLonLatAlt, buildLocalFrame } from './coord.js';

const TILESET_URL = 'https://data.mars3d.cn/3dtiles/qx-simiao/tileset.json';
const DEFAULT_BBOX_SIZE = 30; // meters

// State management
class App {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.tileset = null;
    this.multiView = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.entities = [];     // list of extracted entities
    this.entityId = 0;
    this.modelReady = false;
    this.lastMergedGeom = null;  // current local mesh geometry (for re-extraction)
    this.lastCenterEcef = null;
    this.lastBboxSize = DEFAULT_BBOX_SIZE;
    this.busy = false;
    this.settings = {
      bboxSize: DEFAULT_BBOX_SIZE,
      viewCount: 5,
      resolution: 512,
      fallbackEnabled: true,
      precision: 'speed'
    };
  }

  async init() {
    // Logger
    logger.attach(
      document.getElementById('diag-body'),
      document.getElementById('diag-header'),
      document.getElementById('diag-panel')
    );
    logger.show();

    // Setup three.js
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e14);
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      50000
    );
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1;
    controls.maxDistance = 5000;
    controls.screenSpacePanning = true;
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.controls = controls;

    // Subtle ambient + directional light
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(0, 0, 1);
    scene.add(dir);

    window.addEventListener('resize', () => this.onResize());

    // Click handler
    renderer.domElement.addEventListener('click', (e) => this.onClick(e));
    renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

    // Load tileset
    this.tileset = new Tileset();
    try {
      await this.tileset.load(TILESET_URL);
    } catch (err) {
      logger.err(`加载 tileset 失败: ${err.message}`);
      throw err;
    }
    scene.add(this.tileset.group);

    // Frame camera to tileset bounding sphere
    const sphere = this.tileset.root.root.boundingVolume.sphere;
    const center = new THREE.Vector3(sphere[0], sphere[1], sphere[2]);
    const radius = sphere[3];
    const frame = this.computeFrameFromEcefCenter(center);
    camera.position.set(
      center.x + radius * 0.6,
      center.y - radius * 0.6,
      center.z + radius * 0.6
    );
    camera.up.set(0, 0, 1);
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();

    // Preload tiles within view
    await this.preloadInitialTiles();

    // Multi-view renderer
    this.multiView = new MultiViewRenderer(renderer);

    // Start rendering loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Setup UI
    this.setupUI();

    // Preload AI model in the background
    this.loadModelInBackground();
  }

  computeFrameFromEcefCenter(center) {
    // Use a simple up = +Z frame; for the demo data this is approximately correct.
    return buildLocalFrame(
      ecefToLonLatAlt(center.x, center.y, center.z).lon,
      ecefToLonLatAlt(center.x, center.y, center.z).lat,
      ecefToLonLatAlt(center.x, center.y, center.z).alt
    );
  }

  async preloadInitialTiles() {
    // Load the root-level tile content for an initial view
    const sphere = this.tileset.root.root.boundingVolume.sphere;
    const queryBox = new THREE.Box3(
      new THREE.Vector3(sphere[0] - sphere[3], sphere[1] - sphere[3], sphere[2] - sphere[3]),
      new THREE.Vector3(sphere[0] + sphere[3], sphere[1] + sphere[3], sphere[2] + sphere[3])
    );
    try {
      const meshes = await this.tileset.getMeshesInBox(queryBox);
      logger.ok(`初始瓦片加载完成: ${meshes.length} 个网格`);
    } catch (err) {
      logger.err(`初始瓦片加载失败: ${err.message}`);
    }
  }

  onResize() {
    const container = document.getElementById('canvas-container');
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  setupUI() {
    // Settings button
    document.getElementById('settings-btn').addEventListener('click', () => {
      const p = document.getElementById('settings-panel');
      p.classList.toggle('hidden');
    });
    document.getElementById('close-settings').addEventListener('click', () => {
      document.getElementById('settings-panel').classList.add('hidden');
    });
    document.getElementById('bbox-size').addEventListener('change', (e) => {
      this.settings.bboxSize = parseFloat(e.target.value);
    });
    document.getElementById('view-count').addEventListener('change', (e) => {
      this.settings.viewCount = parseInt(e.target.value, 10);
    });
    document.getElementById('render-res').addEventListener('change', (e) => {
      this.settings.resolution = parseInt(e.target.value, 10);
    });
    document.getElementById('fallback-enabled').addEventListener('change', (e) => {
      this.settings.fallbackEnabled = e.target.checked;
    });
    document.getElementById('retry-model').addEventListener('click', () => {
      this.loadModelInBackground();
    });

    // Property panel
    document.getElementById('close-property').addEventListener('click', () => {
      document.getElementById('property-panel').classList.add('hidden');
    });
    document.getElementById('export-glb').addEventListener('click', () => {
      if (this.selectedEntity) {
        exportGlb(this.selectedEntity.geometry, `entity_${this.selectedEntity.id}.glb`)
          .then(() => logger.ok('GLB 导出完成'))
          .catch((e) => logger.err(`GLB 导出失败: ${e.message}`));
      }
    });
    document.getElementById('export-geojson').addEventListener('click', () => {
      if (this.selectedEntity) {
        try {
          const props = exportGeoJSON(this.selectedEntity.geometry, `entity_${this.selectedEntity.id}.geojson`);
          logger.ok(`GeoJSON 导出完成，面积 ${props.area_m2.toFixed(1)} m²，高度 ${props.height_m.toFixed(1)} m`);
        } catch (err) {
          logger.err(`GeoJSON 导出失败: ${err.message}`);
        }
      }
    });
    document.getElementById('delete-entity').addEventListener('click', () => {
      if (this.selectedEntity) {
        this.deleteEntity(this.selectedEntity.id);
      }
    });
  }

  async loadModelInBackground() {
    const banner = document.getElementById('loading-banner');
    const fill = document.getElementById('progress-fill');
    const detail = document.getElementById('loading-detail');
    banner.classList.remove('hidden');
    try {
      const totalBytes = getModelSize();
      let downloaded = 0;
      // Simulate progress for visual feedback (model is loaded via fetch
      // inside onnxruntime-web, but we don't have direct progress events
      // for that). For a single 4.4 MB file on a fast link, the actual
      // download is sub-second; we show a brief progress animation.
      const animateProgress = () => new Promise((resolve) => {
        let v = 0;
        const id = setInterval(() => {
          v = Math.min(95, v + 5 + Math.random() * 8);
          fill.style.width = v + '%';
          detail.textContent = `正在下载模型 (${(v * totalBytes / 100 / 1024).toFixed(0)} / ${(totalBytes / 1024).toFixed(0)} KB)`;
          if (v >= 95) { clearInterval(id); resolve(); }
        }, 80);
      });
      await Promise.all([
        getSegmentationModel().catch((e) => { throw e; }),
        animateProgress()
      ]);
      fill.style.width = '100%';
      detail.textContent = '模型加载完成';
      this.modelReady = true;
      banner.classList.add('hidden');
      document.getElementById('ready-status').classList.remove('hidden');
      logger.ok('AI 模型就绪，可以开始分割');
    } catch (err) {
      logger.err(`模型加载失败: ${err.message}`);
      detail.textContent = `模型加载失败: ${err.message}。可以重试或继续使用几何降级方案。`;
      this.modelReady = false;
    }
  }

  async onClick(event) {
    if (this.busy) return;
    // Use left click only
    if (event.button !== 0) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);

    // Raycast against the entire tileset
    const intersects = this.raycaster.intersectObjects(this.tileset.group.children, true);
    if (intersects.length === 0) {
      logger.info('未命中任何几何');
      return;
    }
    const hit = intersects[0];
    const point = hit.point;
    logger.info(`点击命中: ECEF(${point.x.toFixed(1)}, ${point.y.toFixed(1)}, ${point.z.toFixed(1)})`);

    this.busy = true;
    const indicator = document.getElementById('ai-indicator');
    indicator.classList.remove('hidden');
    const t0 = performance.now();

    try {
      // Build a bounding box centered on the hit point, in local frame
      const halfSize = this.settings.bboxSize / 2;
      const lla = ecefToLonLatAlt(point.x, point.y, point.z);
      const frame = buildLocalFrame(lla.lon, lla.lat, lla.alt);
      // The query box is built in the ECEF frame: extend +/- halfSize in
      // each of the local frame axes (e, n, u).
      const corners = [
        [-halfSize, -halfSize, -halfSize],
        [ halfSize, -halfSize, -halfSize],
        [-halfSize,  halfSize, -halfSize],
        [ halfSize,  halfSize, -halfSize],
        [-halfSize, -halfSize,  halfSize],
        [ halfSize, -halfSize,  halfSize],
        [-halfSize,  halfSize,  halfSize],
        [ halfSize,  halfSize,  halfSize]
      ];
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const [e, n, u] of corners) {
        const ec = frame.toECEF(e, n, u);
        minX = Math.min(minX, ec.x); maxX = Math.max(maxX, ec.x);
        minY = Math.min(minY, ec.y); maxY = Math.max(maxY, ec.y);
        minZ = Math.min(minZ, ec.z); maxZ = Math.max(maxZ, ec.z);
      }
      const queryBox = new THREE.Box3(
        new THREE.Vector3(minX, minY, minZ),
        new THREE.Vector3(maxX, maxY, maxZ)
      );

      // Get meshes in the box
      const t1 = performance.now();
      const meshes = await this.tileset.getMeshesInBox(queryBox);
      const dtFetch = performance.now() - t1;
      if (meshes.length === 0) {
        logger.warn('包围盒内没有可用的瓦片');
        return;
      }
      logger.time(`[步骤 1] 局部数据获取 ${meshes.length} 个网格: ${dtFetch.toFixed(0)} ms`);

      // Build merged geometry
      const t2 = performance.now();
      const merged = this.tileset.buildMergedGeometry(meshes);
      const dtMerge = performance.now() - t2;
      logger.time(`[步骤 2] 合并几何体: ${dtMerge.toFixed(0)} ms, 顶点数 ${merged.attributes.position.count}`);

      // Filter to triangles that are within the local box (some
      // triangles from neighboring tiles may have leaked in due to
      // the spherical bounding-volume check).
      this.clipToBox(merged, queryBox);

      // Compute center of the local mesh in ECEF (use the click point
      // as the "look-at" center for multi-view rendering).
      const centerEcef = point.clone();

      // Multi-view rendering
      const t3 = performance.now();
      const views = this.multiView.render(
        new THREE.Mesh(merged, new THREE.MeshBasicMaterial({ vertexColors: true })),
        centerEcef,
        this.settings.bboxSize,
        { viewCount: this.settings.viewCount, resolution: this.settings.resolution }
      );
      const dtRender = performance.now() - t3;
      logger.time(`[步骤 3] 多视角离屏渲染 ${views.length} 张图: ${dtRender.toFixed(0)} ms`);

      // Save for debugging / re-extraction
      this.lastMergedGeom = merged;
      this.lastCenterEcef = centerEcef;
      this.lastBboxSize = this.settings.bboxSize;

      // AI inference
      const t4 = performance.now();
      let segResults = null;
      let usedFallback = false;
      if (this.modelReady) {
        try {
          const timeoutMs = 4000;
          segResults = await Promise.race([
            inferBatch(views),
            new Promise((_, rej) => setTimeout(() => rej(new Error('推理超时')), timeoutMs))
          ]);
        } catch (err) {
          logger.warn(`AI 推理失败: ${err.message}，切换到几何降级方案`);
          this.modelReady = false;
        }
      }
      if (!segResults) {
        if (this.settings.fallbackEnabled) {
          usedFallback = true;
          // Build a synthetic "labels" array per view from the
          // geometric region-growing result. We do this in the next step.
          segResults = views.map((v) => ({
            name: v.name,
            width: v.width,
            height: v.height,
            labels: new Int32Array(v.width * v.height),
            count: 0
          }));
        } else {
          throw new Error('AI 推理失败且未启用降级');
        }
      }
      const dtInfer = performance.now() - t4;
      logger.time(`[步骤 4] 全部推理: ${dtInfer.toFixed(0)} ms`);

      // 2D -> 3D voting
      const t5 = performance.now();
      // For fallback, we run region growing on the merged mesh (only
      // the click triangle is the seed) and assign each triangle the
      // same label (1) so that the voting picks it up.
      let triLabels = null;
      let triCentroids = null;
      if (usedFallback) {
        const refTi = findClickTriangle(merged, new Float32Array(merged.attributes.position.count), point);
        const labels = regionGrow(merged, refTi, { maxFaces: 50000 });
        triLabels = labels;
        // Build centroids once
        const triCount = labels.length;
        triCentroids = new Float32Array(triCount * 3);
        const pos = merged.attributes.position;
        const idx = merged.index;
        const v = new THREE.Vector3();
        for (let ti = 0; ti < triCount; ti++) {
          const i0 = idx ? idx.getX(ti * 3) : ti * 3;
          const i1 = idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1;
          const i2 = idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2;
          v.fromBufferAttribute(pos, i0);
          triCentroids[ti * 3]     = v.x;
          triCentroids[ti * 3 + 1] = v.y;
          triCentroids[ti * 3 + 2] = v.z;
          v.fromBufferAttribute(pos, i1);
          triCentroids[ti * 3]     += v.x;
          triCentroids[ti * 3 + 1] += v.y;
          triCentroids[ti * 3 + 2] += v.z;
          v.fromBufferAttribute(pos, i2);
          triCentroids[ti * 3]     = (triCentroids[ti * 3] + v.x) / 3;
          triCentroids[ti * 3 + 1] = (triCentroids[ti * 3 + 1] + v.y) / 3;
          triCentroids[ti * 3 + 2] = (triCentroids[ti * 3 + 2] + v.z) / 3;
        }
      } else {
        // Combine segResults with the views so voting can find the
        // labels (it needs view.labels)
        const viewsWithLabels = views.map((v, i) => ({
          camera: segResults[i].camera || v.camera,
          width: v.width,
          height: v.height,
          labels: segResults[i].labels
        }));
        const voted = voteTriangles(merged, viewsWithLabels);
        triLabels = voted.triLabels;
        triCentroids = voted.triCentroids;
      }
      const dtVote = performance.now() - t5;
      logger.time(`[步骤 5] 投影与投票: ${dtVote.toFixed(0)} ms`);

      // Find the triangle closest to the click
      const refTi = findClickTriangle(merged, triCentroids, point);
      if (refTi < 0) {
        logger.warn('未找到点击附近的三角面');
        return;
      }
      const refLabel = triLabels[refTi];
      logger.info(`参考三角面 #${refTi}，实例 ID = ${refLabel}`);

      // Extract entity triangles
      const t6 = performance.now();
      const entityTris = extractEntityTriangles(triLabels, refLabel, refTi, triCentroids, 8);
      const dtExtract = performance.now() - t6;
      if (entityTris.length === 0) {
        logger.warn('投票结果为空，请尝试其他位置或调整包围盒');
        return;
      }
      logger.time(`[步骤 6] 提取实体: ${dtExtract.toFixed(0)} ms, 三角面 ${entityTris.length}`);

      // Build the entity geometry
      const entityGeom = buildEntityGeometry(merged, entityTris);

      // Compute center in lon/lat/alt
      const llaCenter = ecefToLonLatAlt(
        entityGeom.boundingSphere?.center.x ?? 0,
        entityGeom.boundingSphere?.center.y ?? 0,
        entityGeom.boundingSphere?.center.z ?? 0
      );
      if (!entityGeom.boundingSphere) entityGeom.computeBoundingSphere();
      const bs = entityGeom.boundingSphere;

      // Add entity to the scene and the list
      const id = ++this.entityId;
      const entity = {
        id,
        geometry: entityGeom,
        mesh: null,
        bboxSize: this.settings.bboxSize,
        area: 0,
        height: 0,
        center: bs.center.clone(),
        lla: llaCenter,
        confidence: usedFallback ? null : (segResults ? this.estimateConfidence(segResults, views, refLabel) : null),
        source: usedFallback ? 'geometry' : 'ai',
        visible: true,
        class: 'building'
      };
      this.computeEntityProperties(entity);
      this.addEntityMesh(entity);
      this.entities.push(entity);
      this.updateEntityList();
      this.selectEntity(id);

      const dtTotal = performance.now() - t0;
      logger.ok(`✓ 实体提取完成，总耗时 ${dtTotal.toFixed(0)} ms，${entityTris.length} 三角面，面积 ${entity.area.toFixed(1)} m²`);
    } catch (err) {
      logger.err(`处理失败: ${err.message}`);
      console.error(err);
    } finally {
      this.busy = false;
      document.getElementById('ai-indicator').classList.add('hidden');
    }
  }

  // Trim a merged geometry so only triangles within queryBox remain.
  clipToBox(geom, queryBox) {
    const pos = geom.attributes.position;
    const idx = geom.index;
    const triCount = idx ? idx.count / 3 : pos.count / 3;
    const keep = [];
    const v = new THREE.Vector3();
    for (let ti = 0; ti < triCount; ti++) {
      const i0 = idx ? idx.getX(ti * 3) : ti * 3;
      const i1 = idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1;
      const i2 = idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2;
      let inside = true;
      for (const i of [i0, i1, i2]) {
        v.fromBufferAttribute(pos, i);
        if (!queryBox.containsPoint(v)) { inside = false; break; }
      }
      if (inside) keep.push(i0, i1, i2);
    }
    geom.setIndex(keep);
    // Re-derive count
    logger.info(`裁剪后: ${keep.length / 3} 三角面 (原 ${triCount})`);
  }

  getViewCamera(centerEcef, name, bboxSize, idx) {
    // Recreate the camera for view name 'idx' (matches multi-view-renderer).
    const distance = bboxSize * 1.2;
    let pitch, yaw;
    if (name === 'top' || idx === 0) { pitch = -89.5; yaw = 0; }
    else if (name === 'n' || idx === 1) { pitch = -45; yaw = 0; }
    else if (name === 'e' || idx === 2) { pitch = -45; yaw = 90; }
    else if (name === 's' || idx === 3) { pitch = -45; yaw = 180; }
    else if (name === 'w' || idx === 4) { pitch = -45; yaw = 270; }
    else { pitch = -45; yaw = (idx - 1) * 72; }
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, distance * 4);
    const pitchRad = (pitch * Math.PI) / 180;
    const yawRad = (yaw * Math.PI) / 180;
    cam.position.set(
      centerEcef.x + distance * Math.cos(pitchRad) * Math.cos(yawRad),
      centerEcef.y + distance * Math.cos(pitchRad) * Math.sin(yawRad),
      centerEcef.z + distance * Math.sin(pitchRad)
    );
    cam.up.set(0, 0, 1);
    cam.lookAt(centerEcef.x, centerEcef.y, centerEcef.z);
    cam.updateMatrixWorld();
    return cam;
  }

  estimateConfidence(segResults, views, refLabel) {
    // Simple: compute the mean mask coverage of the chosen instance.
    let total = 0, matching = 0;
    for (const r of segResults) {
      for (let i = 0; i < r.labels.length; i++) {
        total++;
        if (r.labels[i] === refLabel) matching++;
      }
    }
    if (total === 0) return 0;
    return matching / total;
  }

  computeEntityProperties(entity) {
    // Compute height, area, and center from geometry
    const geom = entity.geometry;
    const pos = geom.attributes.position;
    const idx = geom.index;
    const triCount = idx ? idx.count / 3 : pos.count / 3;
    let minZ = Infinity, maxZ = -Infinity;
    const cx = entity.center.x, cy = entity.center.y, cz = entity.center.z;
    const lla = ecefToLonLatAlt(cx, cy, cz);
    const frame = buildLocalFrame(lla.lon, lla.lat, lla.alt);
    const localPts = [];
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
      const lp = frame.toLocal(x, y, z);
      localPts.push([lp.e, lp.n]);
    }
    entity.height = maxZ - minZ;
    const hull = alphaShape2D(localPts, 0.5);
    if (hull.length >= 3) {
      entity.area = polygonArea(hull);
    } else {
      // Fallback: sum of triangle areas
      let s = 0;
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      for (let ti = 0; ti < triCount; ti++) {
        const i0 = idx ? idx.getX(ti * 3) : ti * 3;
        const i1 = idx ? idx.getX(ti * 3 + 1) : ti * 3 + 1;
        const i2 = idx ? idx.getX(ti * 3 + 2) : ti * 3 + 2;
        a.fromBufferAttribute(pos, i0);
        b.fromBufferAttribute(pos, i1);
        c.fromBufferAttribute(pos, i2);
        const ab = new THREE.Vector3().subVectors(b, a);
        const ac = new THREE.Vector3().subVectors(c, a);
        s += ab.cross(ac).length() / 2;
      }
      entity.area = s;
    }
    entity.minZ = minZ;
    entity.maxZ = maxZ;
    entity.localFrame = frame;
  }

  addEntityMesh(entity) {
    const material = new THREE.MeshBasicMaterial({
      color: 0x4f8cff,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthTest: false
    });
    const mesh = new THREE.Mesh(entity.geometry, material);
    mesh.renderOrder = 999;
    mesh.userData.entityId = entity.id;
    this.scene.add(mesh);
    entity.mesh = mesh;
    // Outline
    const edges = new THREE.EdgesGeometry(entity.geometry, 30);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x4fc3ff, transparent: true, opacity: 0.9 });
    const lines = new THREE.LineSegments(edges, lineMat);
    lines.renderOrder = 1000;
    mesh.add(lines);
    entity.outline = lines;
  }

  updateEntityList() {
    const list = document.getElementById('entity-list-body');
    list.innerHTML = '';
    document.getElementById('entity-count').textContent = this.entities.length;
    document.getElementById('entity-list').classList.remove('hidden');
    for (const e of this.entities) {
      const item = document.createElement('div');
      item.className = 'entity-item' + (this.selectedEntity?.id === e.id ? ' active' : '');
      item.innerHTML = `
        <div class="entity-item-info">
          <div class="entity-item-name">#${e.id} ${this.classLabel(e.class)}</div>
          <div class="entity-item-meta">${e.area.toFixed(1)} m² · ${e.height.toFixed(1)} m · ${e.source === 'ai' ? 'AI' : '几何'}</div>
        </div>
        <div class="entity-item-actions">
          <button class="entity-action-btn" data-action="toggle" title="显隐">${e.visible ? '●' : '○'}</button>
          <button class="entity-action-btn" data-action="delete" title="删除">×</button>
        </div>
      `;
      item.addEventListener('click', (ev) => {
        if (ev.target.dataset.action === 'toggle') {
          this.toggleEntity(e.id);
        } else if (ev.target.dataset.action === 'delete') {
          this.deleteEntity(e.id);
        } else {
          this.selectEntity(e.id);
        }
      });
      list.appendChild(item);
    }
  }

  classLabel(c) {
    const map = {
      building: '建筑',
      tree: '树木',
      road: '道路',
      ground: '地面',
      vehicle: '车辆',
      other: '其他'
    };
    return map[c] || c;
  }

  selectEntity(id) {
    this.selectedEntity = this.entities.find((e) => e.id === id);
    if (!this.selectedEntity) return;
    document.getElementById('property-panel').classList.remove('hidden');
    const e = this.selectedEntity;
    document.getElementById('prop-id').textContent = `#${e.id}`;
    document.getElementById('prop-lon').textContent = e.lla.lon.toFixed(6) + '°';
    document.getElementById('prop-lat').textContent = e.lla.lat.toFixed(6) + '°';
    document.getElementById('prop-elev').textContent = e.lla.alt.toFixed(1);
    document.getElementById('prop-area').textContent = e.area.toFixed(1);
    document.getElementById('prop-height').textContent = e.height.toFixed(2);
    document.getElementById('prop-vertex').textContent = e.geometry.attributes.position.count;
    document.getElementById('prop-faces').textContent = (e.geometry.index?.count || e.geometry.attributes.position.count) / 3;
    document.getElementById('prop-conf').textContent = e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : 'N/A';
    const select = document.getElementById('prop-class');
    select.innerHTML = '';
    for (const [k, v] of Object.entries({ building: '建筑', tree: '树木', road: '道路', ground: '地面', vehicle: '车辆', other: '其他' })) {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = v;
      if (k === e.class) opt.selected = true;
      select.appendChild(opt);
    }
    select.onchange = () => { e.class = select.value; this.updateEntityList(); };
    this.updateEntityList();
  }

  toggleEntity(id) {
    const e = this.entities.find((e) => e.id === id);
    if (!e) return;
    e.visible = !e.visible;
    e.mesh.visible = e.visible;
    this.updateEntityList();
  }

  deleteEntity(id) {
    const idx = this.entities.findIndex((e) => e.id === id);
    if (idx < 0) return;
    const e = this.entities[idx];
    this.scene.remove(e.mesh);
    e.mesh.geometry.dispose();
    e.mesh.material.dispose();
    if (e.outline) {
      e.outline.geometry.dispose();
      e.outline.material.dispose();
    }
    this.entities.splice(idx, 1);
    if (this.selectedEntity?.id === id) {
      this.selectedEntity = null;
      document.getElementById('property-panel').classList.add('hidden');
    }
    this.updateEntityList();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  window.__app = app;
  app.init().catch((err) => {
    logger.err(`应用初始化失败: ${err.message}`);
    console.error(err);
  });
});
