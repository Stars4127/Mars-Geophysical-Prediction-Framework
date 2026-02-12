const southWest = L.latLng(-90, -180),
    northEast = L.latLng(90, 180),
    bounds = L.latLngBounds(southWest, northEast);

const map = L.map('map', {
    crs: L.CRS.EPSG4326,
    center: [0, 0],
    zoom: 2,
    minZoom: 0,
    maxZoom: 6,
    maxBounds: bounds,
    maxBoundsViscosity: 1.0,
    attributionControl: false,
    zoomSnap: 0.1,
    zoomControl: false // Disable default zoom control to reposition
});

// Add zoom control at topright
L.control.zoom({ position: 'topright' }).addTo(map);

// Sidebar logic
const sidebar = document.getElementById('sidebar');
const toggleBtn = document.getElementById('toggle-sidebar');
const closeBtn = document.getElementById('close-sidebar');
const recalculateBtn = document.getElementById('recalculate-btn');
const opacitySlider = document.getElementById('slider-opacity');
const opacityVal = document.getElementById('val-opacity');

toggleBtn.addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
    toggleBtn.classList.add('hidden');
});

closeBtn.addEventListener('click', () => {
    sidebar.classList.add('collapsed');
    toggleBtn.classList.remove('hidden');
});

// Initialize collapsed
sidebar.classList.add('collapsed');

// Function to fit map
function fitMapToScreen() {
    const container = map.getSize();
    const minZoomH = Math.log2(container.y / 256);
    const minZoomW = Math.log2(container.x / 512);
    const fitZoom = Math.max(minZoomH, minZoomW);
    map.setMinZoom(fitZoom);
    if (map.getZoom() < fitZoom) map.setZoom(fitZoom);
}

fitMapToScreen();
window.addEventListener('resize', fitMapToScreen);

// Base Map Layer (Local Proxy)
const baseLayer = L.tileLayer('http://127.0.0.1:8000/tiles/{z}/{x}/{y}.jpg', {
    maxZoom: 12,
    maxNativeZoom: 6,
    minZoom: 0,
    noWrap: true,
    bounds: bounds,
    tileSize: 256,
    className: 'mars-tiles'
}).addTo(map);

// Heatmap Layer
let currentHeatmap = null;

function updateHeatmap(force = false) {
    const weights = {
        water: document.getElementById('weight-water').value,
        dust: document.getElementById('weight-dust').value,
        temp: document.getElementById('weight-temp').value,
        temp_var: document.getElementById('weight-temp-var').value,
        gravity: document.getElementById('weight-gravity').value,
        ti_var: document.getElementById('weight-ti-var').value
    };

    // Update UI values
    document.getElementById('val-water').innerText = weights.water;
    document.getElementById('val-dust').innerText = weights.dust;
    document.getElementById('val-temp').innerText = weights.temp;
    document.getElementById('val-temp-var').innerText = weights.temp_var;
    document.getElementById('val-gravity').innerText = weights.gravity;
    document.getElementById('val-ti-var').innerText = weights.ti_var;

    if (force) {
        const loadingOverlay = document.getElementById('loading-overlay');
        const url = `http://127.0.0.1:8000/api/v1/heatmap/{z}/{x}/{y}.png?water=${weights.water}&dust=${weights.dust}&temp=${weights.temp}&temp_var=${weights.temp_var}&gravity=${weights.gravity}&ti_var=${weights.ti_var}&t=${Date.now()}`;

        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        recalculateBtn.classList.add('loading');
        recalculateBtn.innerText = 'Calculating...';

        if (currentHeatmap) {
            map.removeLayer(currentHeatmap);
        }

        currentHeatmap = L.tileLayer(url, {
            maxZoom: 12,
            maxNativeZoom: 6,
            minZoom: 0,
            noWrap: true,
            bounds: bounds,
            tileSize: 256,
            opacity: parseFloat(opacitySlider.value),
            zIndex: 10
        }).addTo(map);

        // Hide loading overlay when tiles start loading
        let tilesLoaded = 0;
        let tilesLoading = 0;
        
        currentHeatmap.on('tileloadstart', () => {
            tilesLoading++;
        });
        
        currentHeatmap.on('tileload', () => {
            tilesLoaded++;
            if (tilesLoaded >= Math.min(tilesLoading, 3)) {
                loadingOverlay.classList.add('hidden');
                recalculateBtn.classList.remove('loading');
                recalculateBtn.innerText = 'Recalculate Heatmap';
            }
        });
        
        currentHeatmap.on('tileerror', () => {
            tilesLoaded++;
            if (tilesLoaded >= Math.min(tilesLoading, 3)) {
                loadingOverlay.classList.add('hidden');
                recalculateBtn.classList.remove('loading');
                recalculateBtn.innerText = 'Recalculate Heatmap';
            }
        });
        
        // Fallback timeout
        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
            recalculateBtn.classList.remove('loading');
            recalculateBtn.innerText = 'Recalculate Heatmap';
        }, 5000);
    }
}

// Opacity control logic
opacitySlider.addEventListener('input', () => {
    const val = opacitySlider.value;
    opacityVal.innerText = val;
    if (currentHeatmap) {
        currentHeatmap.setOpacity(parseFloat(val));
    }
});

// Event listeners for sliders - just update values, don't trigger tile refresh immediately for performance
const sliders = ['water', 'dust', 'temp', 'temp-var', 'gravity', 'ti-var'];
sliders.forEach(id => {
    document.getElementById(`weight-${id}`).addEventListener('input', () => updateHeatmap(false));
});

recalculateBtn.addEventListener('click', () => updateHeatmap(true));

// Initialize first heatmap
updateHeatmap(true);

L.control.scale().addTo(map);
