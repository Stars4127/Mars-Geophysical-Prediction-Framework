// Bounds for Mars
const southWest = L.latLng(-90, -180),
    northEast = L.latLng(90, 180),
    bounds = L.latLngBounds(southWest, northEast);

// Initialize Map
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
    zoomControl: false // Using custom top-right zoom
});

// Custom Premium Zoom
L.control.zoom({ position: 'topright' }).addTo(map);

// Fit map to screen logic
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

// Heatmap Layer State
let currentHeatmap = null;

function updateSliderLabels() {
    const weights = {
        water: document.getElementById('weight-water').value,
        dust: document.getElementById('weight-dust').value,
        temp: document.getElementById('weight-temp').value,
        temp_var: document.getElementById('weight-temp-var').value,
        gravity: document.getElementById('weight-gravity').value,
        ti_var: document.getElementById('weight-ti-var').value
    };

    // Update UI values ONLY
    document.getElementById('val-water').innerText = weights.water;
    document.getElementById('val-dust').innerText = weights.dust;
    document.getElementById('val-temp').innerText = weights.temp;
    document.getElementById('val-temp-var').innerText = weights.temp_var;
    document.getElementById('val-gravity').innerText = weights.gravity;
    document.getElementById('val-ti-var').innerText = weights.ti_var;

    return weights;
}

function refreshHeatmap() {
    const weights = updateSliderLabels();
    const url = `http://127.0.0.1:8000/api/v1/heatmap/{z}/{x}/{y}.png?water=${weights.water}&dust=${weights.dust}&temp=${weights.temp}&temp_var=${weights.temp_var}&gravity=${weights.gravity}&ti_var=${weights.ti_var}`;

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
        opacity: 0.75,
        zIndex: 10
    }).addTo(map);

    // Sync visibility toggle
    if (!document.getElementById('toggle-heatmap').checked) {
        map.removeLayer(currentHeatmap);
    }
}

// Sidebar Management
function openSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggle-sidebar');
    sidebar.classList.remove('collapsed');
    toggleBtn.classList.add('hidden');

    // Resize map after transition
    setTimeout(() => map.invalidateSize(), 350);
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggle-sidebar');
    sidebar.classList.add('collapsed');
    toggleBtn.classList.remove('hidden');

    // Resize map after transition
    setTimeout(() => map.invalidateSize(), 350);
}

// Event Listeners
document.getElementById('toggle-sidebar').addEventListener('click', openSidebar);
document.getElementById('close-sidebar').addEventListener('click', closeSidebar);

// Sliders
const sliders = ['water', 'dust', 'temp', 'temp-var', 'gravity', 'ti-var'];
sliders.forEach(id => {
    document.getElementById(`weight-${id}`).addEventListener('input', updateSliderLabels);
});

// Calculate Button
document.getElementById('calculate-btn').addEventListener('click', refreshHeatmap);

// Heatmap Overlay Toggle
document.getElementById('toggle-heatmap').addEventListener('change', (e) => {
    if (currentHeatmap) {
        if (e.target.checked) {
            map.addLayer(currentHeatmap);
        } else {
            map.removeLayer(currentHeatmap);
        }
    }
});

// Keyboard Support
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSidebar();
});

// Init
refreshHeatmap();
L.control.scale().addTo(map);
