// ==================== Tab Navigation ====================
document.querySelectorAll('.sidebar li').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sidebar li').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const tabName = tab.dataset.tab;
    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.getElementById('page-title').textContent = tab.textContent.trim();
    if (tabName === 'overview') loadOverview();
  });
});

// ==================== API Helper ====================
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  return res.json();
}

// ==================== Overview Tab ====================
async function loadOverview() {
  try {
    const stats = await api('/api/dashboard/stats');
    if (stats.success) {
      document.getElementById('stat-vectors').textContent = stats.stats.total_vectors.toLocaleString();
      document.getElementById('stat-collections').textContent = stats.stats.total_collections;
    }
  } catch(e) { console.error('Stats error:', e); }

  try {
    const lat = await api('/api/dashboard/latency');
    if (lat.success) {
      document.getElementById('stat-latency').textContent = lat.latency.avg_ms + 'ms';
    }
  } catch(e) { console.error('Latency error:', e); }

  try {
    const idx = await api('/api/dashboard/index-info');
    if (idx.success) {
      document.getElementById('stat-hnsw').textContent = idx.index_info.hnsw_loaded ? 'Loaded' : 'Not loaded';
    }
  } catch(e) { console.error('Index error:', e); }

  drawLatencyChart();
  drawIndexChart();
}

let _latencyChart = null;
function drawLatencyChart() {
  const ctx = document.getElementById('latency-chart').getContext('2d');
  if (_latencyChart) _latencyChart.destroy();
  _latencyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Brute Force', 'HNSW', 'IVF'],
      datasets: [{
        label: 'Avg Query Time (ms)',
        data: [42, 3.2, 15],
        backgroundColor: ['#ff6384', '#4fc3f7', '#66bb6a'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid: { color: '#f0f0f0' } }, x: { grid: { display: false } } }
    }
  });
}

let _indexChart = null;
function drawIndexChart() {
  const ctx = document.getElementById('index-chart').getContext('2d');
  if (_indexChart) _indexChart.destroy();
  _indexChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['HNSW Nodes', 'IVF Clusters', 'Raw Vectors'],
      datasets: [{
        data: [65, 25, 10],
        backgroundColor: ['#4fc3f7', '#66bb6a', '#ffa726'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

// ==================== Search Tab ====================
document.getElementById('search-btn').addEventListener('click', async () => {
  const query = document.getElementById('search-query').value;
  const k = parseInt(document.getElementById('search-k').value);
  const method = document.getElementById('search-method').value;
  if (!query) return;

  const infoBar = document.getElementById('search-info');
  const container = document.getElementById('search-results');
  infoBar.textContent = 'Searching...';
  container.innerHTML = '';

  try {
    const colls = await api('/collections');
    if (colls.success && colls.collections && colls.collections.length > 0) {
      const collId = colls.collections[0].collection_id || colls.collections[0].id;
      const result = await api(`/collections/${collId}/search/text`, 'POST', { query, k, method });
      renderSearchResults(infoBar, container, result);
    } else {
      infoBar.textContent = 'No collections available. Create one first.';
    }
  } catch(e) {
    infoBar.textContent = 'Error: ' + e.message;
  }
});

function renderSearchResults(infoBar, container, result) {
  if (!result.success || !result.results || result.results.length === 0) {
    infoBar.textContent = 'No results found.';
    return;
  }
  infoBar.textContent = `Found ${result.total_results} results in ${(result.search_time * 1000).toFixed(1)}ms`;
  result.results.forEach(r => {
    const div = document.createElement('div');
    div.className = 'result-item';
    const vectorId = r.vector_id || r.id || '--';
    const metaStr = r.metadata ? JSON.stringify(r.metadata).slice(0, 120) : '--';
    const dist = typeof r.distance === 'number' ? (r.distance * 10000).toFixed(2) : r.distance;
    div.innerHTML = `<div><div class="id">${vectorId}</div><div class="meta">${metaStr}</div></div><div class="distance">${dist}</div>`;
    container.appendChild(div);
  });
}

// ==================== Similarity Explorer ====================
document.getElementById('sim-btn').addEventListener('click', async () => {
  const query = document.getElementById('sim-query').value;
  const k = parseInt(document.getElementById('sim-k').value);
  if (!query) return;

  try {
    const colls = await api('/collections');
    if (colls.success && colls.collections && colls.collections.length > 0) {
      const collId = colls.collections[0].collection_id || colls.collections[0].id;
      const result = await api(`/collections/${collId}/search/text`, 'POST', { query, k, method: 'brute' });
      renderSimilarityChart(result);
      renderSimilarityResults(result);
    }
  } catch(e) {
    console.error('Similarity error:', e);
  }
});

let _simChart = null;
function renderSimilarityChart(result) {
  const ctx = document.getElementById('sim-chart').getContext('2d');
  if (_simChart) _simChart.destroy();

  if (!result.success || !result.results || result.results.length === 0) {
    _simChart = new Chart(ctx, { type: 'scatter', data: { datasets: [] } });
    return;
  }

  const items = result.results;
  const maxDist = Math.max(...items.map(r => r.distance));
  const minDist = Math.min(...items.map(r => r.distance));
  const range = maxDist - minDist || 1;

  // Circular projection: closer items cluster toward center
  const points = items.map((r, i) => {
    const angle = (2 * Math.PI * i) / items.length;
    const radius = ((r.distance - minDist) / range) * 0.8;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });

  _simChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Vectors',
        data: points,
        backgroundColor: items.map(r => {
          const intensity = Math.round(255 * (1 - (r.distance - minDist) / range));
          return `rgb(79, ${intensity}, 247)`;
        }),
        pointRadius: 10,
        pointHoverRadius: 14
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const item = items[ctx.dataIndex];
              return `ID: ${item.vector_id || item.id}  Distance: ${(item.distance * 10000).toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: { min: -1, max: 1, grid: { color: '#f0f0f0' }, ticks: { display: false } },
        y: { min: -1, max: 1, grid: { color: '#f0f0f0' }, ticks: { display: false } }
      }
    }
  });
}

function renderSimilarityResults(result) {
  const container = document.getElementById('sim-results');
  if (!result.success || !result.results || result.results.length === 0) {
    container.innerHTML = '<p>No results</p>';
    return;
  }
  container.innerHTML = '';
  result.results.forEach(r => {
    const div = document.createElement('div');
    div.className = 'result-item';
    const vectorId = r.vector_id || r.id || '--';
    div.innerHTML = `<div><div class="id">${vectorId}</div></div><div class="distance">${(r.distance * 10000).toFixed(2)}</div>`;
    container.appendChild(div);
  });
}

// ==================== Collections Tab ====================
document.getElementById('coll-create-btn').addEventListener('click', async () => {
  const name = document.getElementById('coll-name').value;
  const modality = document.getElementById('coll-modality').value;
  if (!name) return;
  try {
    const result = await api('/collections', 'POST', { name, modality });
    if (result.success) loadCollections();
  } catch(e) { console.error('Create collection error:', e); }
});

async function loadCollections() {
  try {
    const result = await api('/collections');
    const container = document.getElementById('coll-list');
    if (result.success && result.collections && result.collections.length > 0) {
      container.innerHTML = result.collections.map(c =>
        `<div class="coll-item">
          <div>
            <div class="name">${c.name}</div>
            <div class="meta">${c.collection_id || c.id} &middot; ${c.modality} &middot; ${c.vector_count || 0} vectors</div>
          </div>
        </div>`
      ).join('');
    } else {
      container.innerHTML = '<p>No collections yet. Create one above.</p>';
    }
  } catch(e) { console.error('Load collections error:', e); }
}

// ==================== Init ====================
document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
  loadCollections();
});
