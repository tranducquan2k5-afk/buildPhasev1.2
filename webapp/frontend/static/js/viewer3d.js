/**
 * Three.js WebGL 3D Point Cloud & Bounding Box Viewer
 * ===================================================
 * Supports:
 * - 60 FPS point cloud rendering via BufferGeometry
 * - Color modes: By Height (Z) & By LiDAR Intensity
 * - Before / After comparison (Original vs. Augmented)
 * - 3D Wireframe Bounding Boxes (Quaternion-rotated)
 * - Camera presets: Isometric, BEV (Top-down), Driver's perspective
 */

class PointCloudViewer3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) throw new Error(`Container #${containerId} not found.`);

        this.colorMode = 'z'; // 'z' or 'intensity'
        this.displayMode = 'augmented'; // 'augmented', 'original', 'both'
        this.pointSize = 0.12;

        this.originalData = null; // flat array [x0, y0, z0, i0, ...]
        this.augmentedData = null;
        this.boxesData = [];

        this.origCloudObj = null;
        this.augCloudObj = null;
        this.boxesGroup = null;

        this.initScene();
    }

    initScene() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x07090E);

        // LiDAR Sensor coordinate frame: Z is UP, X is forward/right, Y is left/forward
        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
        this.camera.position.set(28, -28, 22);
        this.camera.up.set(0, 0, 1);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0, 0);

        // Ground grid in XY plane
        const grid = new THREE.GridHelper(80, 40, 0x06B6D4, 0x1E293B);
        grid.rotation.x = Math.PI / 2;
        this.scene.add(grid);

        // Axes helper (X: Red, Y: Green, Z: Blue)
        const axes = new THREE.AxesHelper(4);
        this.scene.add(axes);

        this.boxesGroup = new THREE.Group();
        this.scene.add(this.boxesGroup);

        window.addEventListener('resize', () => this.onWindowResize());
        this.animate();
    }

    onWindowResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    setData(previewData) {
        if (!previewData) return;
        this.originalData = previewData.original_points;
        this.augmentedData = previewData.augmented_points;
        this.boxesData = previewData.boxes || [];
        this.rebuildMeshes();
    }

    setColorMode(mode) {
        this.colorMode = mode;
        this.rebuildMeshes();
    }

    setDisplayMode(mode) {
        this.displayMode = mode;
        if (this.origCloudObj) this.origCloudObj.visible = (mode === 'original' || mode === 'both');
        if (this.augCloudObj) this.augCloudObj.visible = (mode === 'augmented' || mode === 'both');
        if (this.boxesGroup) this.boxesGroup.visible = (mode === 'augmented' || mode === 'both');
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.augCloudObj && this.augCloudObj.material) {
            this.augCloudObj.material.size = size;
        }
        if (this.origCloudObj && this.origCloudObj.material) {
            this.origCloudObj.material.size = size;
        }
    }

    setCameraView(viewType) {
        if (viewType === 'iso') {
            this.camera.position.set(28, -28, 22);
            this.controls.target.set(0, 0, 0);
        } else if (viewType === 'bev') {
            this.camera.position.set(0, 0, 55);
            this.controls.target.set(0, 0, 0);
        } else if (viewType === 'driver') {
            this.camera.position.set(-2, 0, 1.8);
            this.controls.target.set(20, 0, 0);
        }
    }

    buildPointsMesh(pointsFlat, isOriginal = false) {
        if (!pointsFlat || pointsFlat.length === 0) return null;
        const numPoints = pointsFlat.length / 4;
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 3);

        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < numPoints; i++) {
            const z = pointsFlat[i * 4 + 2];
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
        const rangeZ = (maxZ - minZ) || 1.0;

        for (let i = 0; i < numPoints; i++) {
            const x = pointsFlat[i * 4 + 0];
            const y = pointsFlat[i * 4 + 1];
            const z = pointsFlat[i * 4 + 2];
            const intensity = pointsFlat[i * 4 + 3];

            positions[i * 3 + 0] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            if (isOriginal) {
                // Dimmed cyan for original
                colors[i * 3 + 0] = 0.3;
                colors[i * 3 + 1] = 0.7;
                colors[i * 3 + 2] = 0.9;
            } else {
                if (this.colorMode === 'z') {
                    const normZ = Math.min(Math.max((z - minZ) / rangeZ, 0.0), 1.0);
                    const c = new THREE.Color();
                    c.setHSL(0.65 - normZ * 0.65, 1.0, 0.5);
                    colors[i * 3 + 0] = c.r;
                    colors[i * 3 + 1] = c.g;
                    colors[i * 3 + 2] = c.b;
                } else {
                    const normI = Math.min(Math.max(intensity / 255.0, 0.05), 1.0);
                    colors[i * 3 + 0] = normI * 0.15;
                    colors[i * 3 + 1] = normI;
                    colors[i * 3 + 2] = normI * 0.95;
                }
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.pointSize,
            vertexColors: true,
            transparent: true,
            opacity: isOriginal ? 0.6 : 0.88
        });

        return new THREE.Points(geometry, material);
    }

    rebuildMeshes() {
        // Clear augmented cloud
        if (this.augCloudObj) {
            this.scene.remove(this.augCloudObj);
            if (this.augCloudObj.geometry) this.augCloudObj.geometry.dispose();
            if (this.augCloudObj.material) this.augCloudObj.material.dispose();
            this.augCloudObj = null;
        }

        // Clear original cloud
        if (this.origCloudObj) {
            this.scene.remove(this.origCloudObj);
            if (this.origCloudObj.geometry) this.origCloudObj.geometry.dispose();
            if (this.origCloudObj.material) this.origCloudObj.material.dispose();
            this.origCloudObj = null;
        }

        // Build augmented
        if (this.augmentedData) {
            this.augCloudObj = this.buildPointsMesh(this.augmentedData, false);
            if (this.augCloudObj) {
                this.augCloudObj.visible = (this.displayMode === 'augmented' || this.displayMode === 'both');
                this.scene.add(this.augCloudObj);
            }
        }

        // Build original
        if (this.originalData) {
            this.origCloudObj = this.buildPointsMesh(this.originalData, true);
            if (this.origCloudObj) {
                this.origCloudObj.visible = (this.displayMode === 'original' || this.displayMode === 'both');
                this.scene.add(this.origCloudObj);
            }
        }

        // Rebuild Bounding Boxes
        this.rebuildBoxes();
    }

    rebuildBoxes() {
        while (this.boxesGroup.children.length > 0) {
            const obj = this.boxesGroup.children[0];
            this.boxesGroup.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }

        if (this.boxesData && this.boxesData.length > 0 && this.displayMode !== 'original') {
            this.boxesData.forEach(b => {
                if (!b || !b.size || !b.translation) return;

                // NuScenes / LiDAR convention: size = [width, length, height] = [w, l, h]
                // Along vehicle local axes:
                // X (heading / forward) = length l
                // Y (lateral / left)   = width w
                // Z (vertical / up)     = height h
                const l = (b.length !== undefined) ? Number(b.length) : Number(b.size[1]);
                const w = (b.width !== undefined) ? Number(b.width) : Number(b.size[0]);
                const h = (b.height !== undefined) ? Number(b.height) : Number(b.size[2]);

                // Three.js BoxGeometry(dx, dy, dz) takes (X_dim, Y_dim, Z_dim) -> (l, w, h)
                const boxGeom = new THREE.BoxGeometry(l, w, h);
                const edgesGeom = new THREE.EdgesGeometry(boxGeom);
                const lineMat = new THREE.LineBasicMaterial({ color: 0xF59E0B, linewidth: 2 });
                const wireframe = new THREE.LineSegments(edgesGeom, lineMat);

                // Geometric center: translation = [tx, ty, tz] is already 3D center (ground_z + h / 2)
                const [tx, ty, tz] = b.translation.map(Number);
                wireframe.position.set(tx, ty, tz);

                // Rotation: nuScenes quaternion [w, x, y, z] -> Three.js (x, y, z, w)
                if (b.rotation && Array.isArray(b.rotation) && b.rotation.length === 4) {
                    wireframe.quaternion.set(b.rotation[1], b.rotation[2], b.rotation[3], b.rotation[0]).normalize();
                } else if (b.yaw !== undefined) {
                    wireframe.rotation.set(0, 0, Number(b.yaw));
                }

                // Visual Indicator 1: Heading Line pointing forward along +X in local frame
                const headingGeom = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    new THREE.Vector3(l / 2 + 0.45, 0, 0)
                ]);
                const headingMat = new THREE.LineBasicMaterial({ color: 0xEF4444, linewidth: 3 });
                const headingLine = new THREE.Line(headingGeom, headingMat);
                wireframe.add(headingLine);

                // Visual Indicator 2: Front face 'X' cross (at +l/2) to clearly identify front of vehicle
                const frontCrossGeom = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(l / 2, -w / 2, -h / 2),
                    new THREE.Vector3(l / 2,  w / 2,  h / 2),
                    new THREE.Vector3(l / 2, -w / 2,  h / 2),
                    new THREE.Vector3(l / 2,  w / 2, -h / 2)
                ]);
                const frontCrossMat = new THREE.LineBasicMaterial({ color: 0xEF4444, linewidth: 2 });
                const frontCross = new THREE.LineSegments(frontCrossGeom, frontCrossMat);
                wireframe.add(frontCross);

                this.boxesGroup.add(wireframe);
            });
        }
    }
}
