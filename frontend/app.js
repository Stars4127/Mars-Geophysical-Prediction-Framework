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
        ti_var: document.getElementById('weight-ti-var').value,
        uncertainty: document.getElementById('weight-uncertainty').value
    };

    // Update UI values
    document.getElementById('val-water').innerText = weights.water;
    document.getElementById('val-dust').innerText = weights.dust;
    document.getElementById('val-temp').innerText = weights.temp;
    document.getElementById('val-temp-var').innerText = weights.temp_var;
    document.getElementById('val-gravity').innerText = weights.gravity;
    document.getElementById('val-ti-var').innerText = weights.ti_var;
    document.getElementById('val-uncertainty').innerText = weights.uncertainty;

    if (force) {
        const loadingOverlay = document.getElementById('loading-overlay');
        const url = `http://127.0.0.1:8000/api/v1/heatmap/{z}/{x}/{y}.png?water=${weights.water}&dust=${weights.dust}&temp=${weights.temp}&temp_var=${weights.temp_var}&gravity=${weights.gravity}&ti_var=${weights.ti_var}&uncertainty=${weights.uncertainty}&t=${Date.now()}`;

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

// Coordinate Tracking
const latDisplay = document.getElementById('lat-display');
const lonDisplay = document.getElementById('lon-display');

map.on('mousemove', (e) => {
    latDisplay.innerText = e.latlng.lat.toFixed(4);
    lonDisplay.innerText = e.latlng.lng.toFixed(4);
});

// Map Click Interaction
map.on('click', async (e) => {
    const { lat, lng } = e.latlng;

    // Create a temporary popup while loading
    const popup = L.popup({
        autoPan: true,
        offset: [0, lat > 70 ? 100 : -10] // Shift down heavily if high polar, else slight shift up
    })
        .setLatLng(e.latlng)
        .setContent('<div class="popup-header"><h3>Analyzing...</h3><p>Retrieving Martian site data</p></div>')
        .openOn(map);

    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/point?lat=${lat}&lon=${lng}`);
        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;

            // Format the content
            let content = `
                <div class="popup-header">
                    <h3>Landing Site Survey</h3>
                    <p>Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}</p>
                </div>
                <div class="popup-data-grid">
                    <div class="popup-data-item">
                        <span class="popup-data-label">Water Content</span>
                        <span class="popup-data-value">${data['Water']?.toFixed(1) || '0.0'}%</span>
                    </div>
                    <div class="popup-data-item">
                        <span class="popup-data-label">Dust Ratio</span>
                        <span class="popup-data-value">${data['Dust']?.toFixed(3) || '0.000'}</span>
                    </div>
                    <div class="popup-data-item">
                        <span class="popup-data-label">Avg Temp</span>
                        <span class="popup-data-value">${data['Surface_Temp']?.toFixed(1) || 'N/A'}°C</span>
                    </div>
                    <div class="popup-data-item">
                        <span class="popup-data-label">Temp Variation</span>
                        <span class="popup-data-value">${data['Temp_Variation']?.toFixed(1) || 'N/A'}°C</span>
                    </div>
                    <div class="popup-data-item">
                        <span class="popup-data-label">Gravity Anom</span>
                        <span class="popup-data-value">${data['Gravity']?.toFixed(1) || '0.0'}</span>
                    </div>
                    <div class="popup-data-item">
                        <span class="popup-data-label">TI Variance</span>
                        <span class="popup-data-value">${data['TI_Variance']?.toFixed(1) || 'N/A'}</span>
                    </div>
                </div>
            `;
            popup.setContent(content);
        } else {
            popup.setContent('<div class="popup-header"><h3>Error</h3><p>Could not retrieve data for this point.</p></div>');
        }
    } catch (err) {
        console.error("Fetch error:", err);
        popup.setContent('<div class="popup-header"><h3>Error</h3><p>Server connection failed.</p></div>');
    }
});

// Event listeners for sliders
const sliders = ['water', 'dust', 'temp', 'temp-var', 'gravity', 'ti-var', 'uncertainty'];
sliders.forEach(id => {
    const el = document.getElementById(`weight-${id}`);
    if (el) el.addEventListener('input', () => updateHeatmap(false));
});

recalculateBtn.addEventListener('click', () => updateHeatmap(true));

updateHeatmap(true);

L.control.scale().addTo(map);
