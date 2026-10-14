/**
 * Application Controller & API Integration
 * =========================================
 * Manages:
 * - Mode 1 (nuScenes Sandbox) & Mode 2 (Custom Upload)
 * - Three.js Viewer synchronization
 * - KPI Benchmark Dashboard integration
 * - File downloads & real-time progress
 */

let viewer = null;
let currentMode = "mode1"; // 'mode1' (nuScenes) or 'mode2' (Upload)
let currentWeather = "none";
let currentWeatherLevel = "moderate";
let selectedUploadFile = null;

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize 3D WebGL Viewer
    viewer = new PointCloudViewer3D('canvas-container');

    // 2. Setup Navigation Tabs (Generator vs. KPI)
    setupMainTabs();

    // 3. Setup Mode Switcher (nuScenes vs. Upload)
    setupModeSwitcher();

    // 4. Setup Controls (Weather, Sliders, Checkboxes)
    setupControlPanels();

    // 5. Setup Viewport Controls (Colors, Camera, Before/After)
    setupViewportControls();

    // 6. Fetch Initial Metadata & KPI
    await loadInitialData();

    // 7. Auto-run initial demo on startup
    setTimeout(() => {
        document.getElementById('btn-run-augment').click();
    }, 500);
});

// Setup Main Top Tabs
function setupMainTabs() {
    const tabGen = document.getElementById('tab-generator');
    const tabKpi = document.getElementById('tab-kpi');
    const viewGen = document.getElementById('view-generator');
    const viewKpi = document.getElementById('view-kpi');

    tabGen.addEventListener('click', () => {
        tabGen.classList.replace('text-slate-400', 'text-cyan-400');
        tabGen.classList.add('border-b-2', 'border-cyan-400');
        tabKpi.classList.replace('text-cyan-400', 'text-slate-400');
        tabKpi.classList.remove('border-b-2', 'border-cyan-400');

        viewGen.classList.remove('hidden');
        viewKpi.classList.add('hidden');
        viewer.onWindowResize();
    });

    tabKpi.addEventListener('click', () => {
        tabKpi.classList.replace('text-slate-400', 'text-cyan-400');
        tabKpi.classList.add('border-b-2', 'border-cyan-400');
        tabGen.classList.replace('text-cyan-400', 'text-slate-400');
        tabGen.classList.remove('border-b-2', 'border-cyan-400');

        viewKpi.classList.remove('hidden');
        viewGen.classList.add('hidden');
        loadKpiDashboard();
    });
}

// Setup Mode 1 vs Mode 2 Switcher
function setupModeSwitcher() {
    const btnMode1 = document.getElementById('btn-select-mode1');
    const btnMode2 = document.getElementById('btn-select-mode2');
    const secMode1 = document.getElementById('sec-mode1-nuscenes');
    const secMode2 = document.getElementById('sec-mode2-upload');

    btnMode1.addEventListener('click', () => {
        currentMode = 'mode1';
        btnMode1.classList.add('bg-cyan-600', 'text-white', 'shadow-lg');
        btnMode1.classList.remove('bg-dark-700', 'text-slate-300');
        btnMode2.classList.remove('bg-cyan-600', 'text-white', 'shadow-lg');
        btnMode2.classList.add('bg-dark-700', 'text-slate-300');

        secMode1.classList.remove('hidden');
        secMode2.classList.add('hidden');
    });

    btnMode2.addEventListener('click', () => {
        currentMode = 'mode2';
        btnMode2.classList.add('bg-cyan-600', 'text-white', 'shadow-lg');
        btnMode2.classList.remove('bg-dark-700', 'text-slate-300');
        btnMode1.classList.remove('bg-cyan-600', 'text-white', 'shadow-lg');
        btnMode1.classList.add('bg-dark-700', 'text-slate-300');

        secMode2.classList.remove('hidden');
        secMode1.classList.add('hidden');
    });

    // File Dropzone for Mode 2
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('upload-file-input');
    const promptEl = document.getElementById('upload-prompt');
    const infoEl = document.getElementById('upload-selected-info');
    const filenameEl = document.getElementById('upload-filename');
    const filesizeEl = document.getElementById('upload-filesize');
    const btnRemove = document.getElementById('upload-btn-remove');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedUploadFile = e.target.files[0];
            filenameEl.textContent = selectedUploadFile.name;
            filesizeEl.textContent = (selectedUploadFile.size / (1024 * 1024)).toFixed(2) + ' MB';
            promptEl.classList.add('hidden');
            infoEl.classList.remove('hidden');
        }
    });

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedUploadFile = null;
        fileInput.value = '';
        promptEl.classList.remove('hidden');
        infoEl.classList.add('hidden');
    });
}

// Setup Control Panels
function setupControlPanels() {
    // Number of objects slider
    const slider = document.getElementById('num-objects-slider');
    const valText = document.getElementById('num-objects-val');
    slider.addEventListener('input', (e) => {
        valText.textContent = e.target.value + ' xe';
    });

    // Weather Tabs
    const wTabs = document.querySelectorAll('.weather-tab-btn');
    const levelContainer = document.getElementById('weather-level-container');
    wTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            wTabs.forEach(t => {
                t.classList.remove('border-cyan-500', 'text-cyan-400');
                t.classList.add('border-dark-500', 'text-slate-300');
            });
            tab.classList.remove('border-dark-500', 'text-slate-300');
            tab.classList.add('border-cyan-500', 'text-cyan-400');
            currentWeather = tab.getAttribute('data-weather');

            if (currentWeather === 'none') {
                levelContainer.classList.add('hidden');
            } else {
                levelContainer.classList.remove('hidden');
            }
        });
    });

    // Weather Level Buttons
    const lvlBtns = document.querySelectorAll('.level-btn');
    lvlBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            lvlBtns.forEach(b => {
                b.classList.remove('border-cyan-500', 'text-cyan-400', 'font-bold');
                b.classList.add('border-dark-500', 'text-slate-300');
            });
            btn.classList.remove('border-dark-500', 'text-slate-300');
            btn.classList.add('border-cyan-500', 'text-cyan-400', 'font-bold');
            currentWeatherLevel = btn.getAttribute('data-level');
        });
    });

    // Run Augment Action
    document.getElementById('btn-run-augment').addEventListener('click', runAugmentation);
}

// Setup Viewport Toolbar Controls
function setupViewportControls() {
    // Color Mode
    const btnColorZ = document.getElementById('btn-color-z');
    const btnColorI = document.getElementById('btn-color-intensity');

    btnColorZ.addEventListener('click', () => {
        viewer.setColorMode('z');
        btnColorZ.classList.replace('bg-dark-700', 'bg-cyan-600');
        btnColorZ.classList.replace('text-slate-300', 'text-white');
        btnColorI.classList.replace('bg-cyan-600', 'bg-dark-700');
        btnColorI.classList.replace('text-white', 'text-slate-300');
    });

    btnColorI.addEventListener('click', () => {
        viewer.setColorMode('intensity');
        btnColorI.classList.replace('bg-dark-700', 'bg-cyan-600');
        btnColorI.classList.replace('text-slate-300', 'text-white');
        btnColorZ.classList.replace('bg-cyan-600', 'bg-dark-700');
        btnColorZ.classList.replace('text-white', 'text-slate-300');
    });

    // Compare Mode (Augmented vs Original vs Split)
    const btnDispAug = document.getElementById('btn-disp-aug');
    const btnDispOrig = document.getElementById('btn-disp-orig');
    const btnDispBoth = document.getElementById('btn-disp-both');

    const updateDispBtns = (activeBtn) => {
        [btnDispAug, btnDispOrig, btnDispBoth].forEach(b => {
            b.classList.remove('bg-cyan-600', 'text-white');
            b.classList.add('bg-dark-700', 'text-slate-300');
        });
        activeBtn.classList.replace('bg-dark-700', 'bg-cyan-600');
        activeBtn.classList.replace('text-slate-300', 'text-white');
    };

    btnDispAug.addEventListener('click', () => {
        viewer.setDisplayMode('augmented');
        updateDispBtns(btnDispAug);
    });
    btnDispOrig.addEventListener('click', () => {
        viewer.setDisplayMode('original');
        updateDispBtns(btnDispOrig);
    });
    btnDispBoth.addEventListener('click', () => {
        viewer.setDisplayMode('both');
        updateDispBtns(btnDispBoth);
    });

    // Camera Views
    document.getElementById('btn-view-iso').addEventListener('click', () => viewer.setCameraView('iso'));
    document.getElementById('btn-view-bev').addEventListener('click', () => viewer.setCameraView('bev'));
    document.getElementById('btn-view-driver').addEventListener('click', () => viewer.setCameraView('driver'));

    // Point size slider
    const ptsSlider = document.getElementById('point-size-slider');
    ptsSlider.addEventListener('input', (e) => {
        viewer.setPointSize(parseFloat(e.target.value));
    });
}

// Fetch Initial Stats & nuScenes Sample Tokens
async function loadInitialData() {
    try {
        const res = await fetch('/api/templates/stats');
        const data = await res.json();
        if (data.status === 'success') {
            const elTplCount = document.getElementById('header-template-count');
            if (elTplCount) elTplCount.textContent = data.total_templates.toLocaleString();
            const elCatCount = document.getElementById('header-cat-count');
            if (elCatCount) elCatCount.textContent = data.total_categories;

            // Populate sample tokens dropdown in Mode 1
            const selectToken = document.getElementById('select-sample-token');
            if (selectToken && data.sample_tokens && data.sample_tokens.length > 0) {
                selectToken.innerHTML = '';
                data.sample_tokens.forEach((tok, idx) => {
                    const opt = document.createElement('option');
                    opt.value = tok;
                    opt.textContent = `Sample ${idx + 1}: ${tok.substring(0, 16)}...`;
                    selectToken.appendChild(opt);
                });
                selectToken.value = data.sample_tokens[0];
            }
        }
    } catch (e) {
        console.warn('Could not load template stats:', e);
    }
}

// Run Augmentation (Mode 1 or Mode 2)
async function runAugmentation() {
    const loading = document.getElementById('loading-overlay');
    loading.classList.remove('hidden');

    const enableInsertion = document.getElementById('toggle-insertion').checked;
    const numObjects = parseInt(document.getElementById('num-objects-slider').value);
    const checkedCats = Array.from(document.querySelectorAll('input[name="cat"]:checked')).map(c => c.value);
    const enableWetRoad = document.getElementById('toggle-wet-road').checked;
    const outputFormat = document.getElementById('export-format').value;

    try {
        let response;
        if (currentMode === 'mode1') {
            // Mode 1: nuScenes Sandbox
            const sampleToken = document.getElementById('select-sample-token').value;
            response = await fetch('/api/augment/nuscenes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sample_token: sampleToken,
                    enable_insertion: enableInsertion,
                    num_objects: numObjects,
                    categories: checkedCats,
                    weather_type: currentWeather,
                    weather_level: currentWeatherLevel,
                    enable_wet_road: enableWetRoad,
                    output_format: outputFormat
                })
            });
        } else {
            // Mode 2: Custom Upload (Mapless Fallback)
            const formData = new FormData();
            if (selectedUploadFile) {
                formData.append('file', selectedUploadFile);
                formData.append('use_sample', 'false');
            } else {
                formData.append('use_sample', 'true');
            }
            formData.append('enable_insertion', enableInsertion);
            formData.append('num_objects', numObjects);
            formData.append('categories', JSON.stringify(checkedCats));
            formData.append('weather_type', currentWeather);
            formData.append('weather_level', currentWeatherLevel);
            formData.append('enable_wet_road', enableWetRoad);
            formData.append('output_format', outputFormat);

            response = await fetch('/api/augment/upload', {
                method: 'POST',
                body: formData
            });
        }

        const res = await response.json();
        if (res.status === 'success') {
            // Update 3D Viewer
            viewer.setData(res.preview);

            // Update Metrics Card
            const stats = res.stats;
            document.getElementById('stat-mode-badge').textContent = stats.mode;
            document.getElementById('stat-initial-pts').textContent = stats.initial_point_count.toLocaleString();
            document.getElementById('stat-final-pts').textContent = stats.final_point_count.toLocaleString();
            document.getElementById('stat-diff-pts').textContent = (stats.point_count_diff >= 0 ? '+' : '') + stats.point_count_diff.toLocaleString();
            document.getElementById('stat-inserted-cnt').textContent = stats.inserted_objects_count + ' xe';
            document.getElementById('stat-time').textContent = stats.elapsed_ms + ' ms';
            document.getElementById('stat-weather-tag').textContent = (stats.weather_type === 'none' ? 'Trời quang' : `${stats.weather_type.toUpperCase()} (${stats.weather_level})`) + (stats.wet_road_enabled ? ' + Vũng nước' : '');

            // Download Links
            document.getElementById('btn-dl-cloud').href = res.download_urls.pointcloud;
            document.getElementById('btn-dl-cloud-tag').textContent = outputFormat.toUpperCase();
            document.getElementById('btn-dl-ann').href = res.download_urls.annotations;
            document.getElementById('btn-dl-zip').href = res.download_urls.zip;

            document.getElementById('stats-card').classList.remove('hidden');
        } else {
            alert('Lỗi: ' + res.detail);
        }
    } catch (err) {
        alert('Lỗi kết nối máy chủ: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

// Load and Render KPI Benchmark Dashboard
async function loadKpiDashboard() {
    try {
        const res = await fetch('/api/kpi');
        const data = await res.json();
        if (data.status === 'success') {
            const h = data.highlights;
            document.getElementById('kpi-valid-rate').textContent = (h.valid_rate * 100).toFixed(0) + '%';
            document.getElementById('kpi-collision').textContent = h.collision_rate.toFixed(1) + '%';
            document.getElementById('kpi-throughput').textContent = h.scenes_per_hour.toLocaleString();
            document.getElementById('kpi-objects').textContent = h.total_benchmark_objects;

            // Fill weather degradation table
            const tbody = document.getElementById('kpi-weather-tbody');
            tbody.innerHTML = '';
            for (const [wName, wData] of Object.entries(data.weather_eval)) {
                const tr = document.createElement('tr');
                tr.className = 'border-b border-dark-600 hover:bg-dark-700/50 transition';
                const ratioPct = (wData.point_ratio * 100).toFixed(1) + '%';
                const diffFmt = wData.point_diff < 0 ? wData.point_diff.toLocaleString() : `+${wData.point_diff}`;
                const origI = wData.intensity_orig_mean.toFixed(1);
                const augI = wData.intensity_aug_mean.toFixed(1);
                const iDrop = ((1 - wData.intensity_aug_mean / (wData.intensity_orig_mean || 1)) * 100).toFixed(1) + '%';

                tr.innerHTML = `
                    <td class="py-3 px-4 font-bold text-slate-200 capitalize">${wName.replace('_', ' ')}</td>
                    <td class="py-3 px-4 font-mono text-cyan-400">${ratioPct} (${diffFmt} pts)</td>
                    <td class="py-3 px-4 font-mono text-slate-300">${origI} &rarr; <span class="text-amber-400 font-bold">${augI}</span></td>
                    <td class="py-3 px-4 font-mono text-red-400 font-semibold">&darr; ${iDrop}</td>
                `;
                tbody.appendChild(tr);
            }
        }
    } catch (e) {
        console.error('Failed to load KPI data:', e);
    }
}

// -------------------------------------------------------------
// PRESET SCENES (100 SCENES) & 3D VISUAL GALLERY EXTENSIONS
// -------------------------------------------------------------

// Load and populate presets list
async function loadPresetsList() {
    try {
        const res = await fetch('/api/presets/list');
        const data = await res.json();
        const sel = document.getElementById('select-preset-scene');
        if (data.status === 'success' && data.presets.length > 0) {
            sel.innerHTML = '';
            data.presets.forEach((p, idx) => {
                const opt = document.createElement('option');
                opt.value = `${p.dataset}:${p.filename}`;
                opt.textContent = `[${p.dataset_name.split(' ')[0]}] ${p.filename} (${p.num_objects} xe)`;
                sel.appendChild(opt);
            });
        } else {
            sel.innerHTML = '<option value="">Không tìm thấy scene có sẵn</option>';
        }
    } catch (e) {
        console.warn('Failed to load presets list:', e);
    }
}

// Setup preset load button
document.addEventListener('DOMContentLoaded', () => {
    loadPresetsList();

    const btnLoadPreset = document.getElementById('btn-load-preset');
    if (btnLoadPreset) {
        btnLoadPreset.addEventListener('click', async () => {
            const selVal = document.getElementById('select-preset-scene').value;
            if (!selVal) return;
            const [dataset, filename] = selVal.split(':');

            const loading = document.getElementById('loading-overlay');
            loading.classList.remove('hidden');

            try {
                const res = await fetch(`/api/presets/load?dataset=${encodeURIComponent(dataset)}&filename=${encodeURIComponent(filename)}`);
                const data = await res.json();
                if (data.status === 'success') {
                    viewer.setData(data.preview);

                    // Update stats
                    const s = data.stats;
                    document.getElementById('stat-mode-badge').textContent = s.mode;
                    document.getElementById('stat-initial-pts').textContent = s.initial_point_count.toLocaleString();
                    document.getElementById('stat-final-pts').textContent = s.final_point_count.toLocaleString();
                    document.getElementById('stat-diff-pts').textContent = '0 (Preset)';
                    document.getElementById('stat-inserted-cnt').textContent = s.inserted_objects_count + ' xe';
                    document.getElementById('stat-time').textContent = s.elapsed_ms + ' ms';
                    document.getElementById('stat-weather-tag').textContent = `${s.weather_type.toUpperCase()} (${s.weather_level})`;

                    document.getElementById('stats-card').classList.remove('hidden');
                } else {
                    alert('Lỗi nạp preset: ' + data.detail);
                }
            } catch (err) {
                alert('Lỗi kết nối khi nạp preset: ' + err.message);
            } finally {
                loading.classList.add('hidden');
            }
        });
    }

    // Vis Gallery Modal
    const btnOpenVis = document.getElementById('btn-open-vis-gallery');
    const visModal = document.getElementById('vis-gallery-modal');
    const btnCloseVis = document.getElementById('btn-close-vis-modal');
    const grid = document.getElementById('vis-gallery-grid');

    if (btnOpenVis && visModal && btnCloseVis) {
        btnOpenVis.addEventListener('click', async () => {
            visModal.classList.remove('hidden');
            grid.innerHTML = '<p class="text-xs text-slate-400 col-span-2 text-center py-8">Đang tải danh sách ảnh 3D...</p>';
            try {
                const res = await fetch('/api/vis/list');
                const data = await res.json();
                if (data.status === 'success' && data.images.length > 0) {
                    grid.innerHTML = '';
                    data.images.forEach(img => {
                        const card = document.createElement('div');
                        card.className = 'bg-dark-900 rounded-xl overflow-hidden border border-dark-600 hover:border-cyan-500 transition shadow-lg group';
                        card.innerHTML = `
                            <a href="${img.url}" target="_blank" class="block relative overflow-hidden">
                                <img src="${img.url}" alt="${img.name}" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
                                <div class="absolute inset-0 bg-gradient-to-t from-dark-950/80 via-transparent to-transparent flex items-end p-3">
                                    <div>
                                        <p class="text-xs font-bold text-white truncate">${img.name}</p>
                                        <p class="text-[10px] text-cyan-400 font-mono">${img.size_kb} KB • Click mở ảnh gốc</p>
                                    </div>
                                </div>
                            </a>
                        `;
                        grid.appendChild(card);
                    });
                } else {
                    grid.innerHTML = '<p class="text-xs text-slate-400 col-span-2 text-center py-8">Chưa có ảnh trong output/vis/</p>';
                }
            } catch (e) {
                grid.innerHTML = '<p class="text-xs text-red-400 col-span-2 text-center py-8">Lỗi tải danh mục ảnh.</p>';
            }
        });

        btnCloseVis.addEventListener('click', () => {
            visModal.classList.add('hidden');
        });
    }
});

