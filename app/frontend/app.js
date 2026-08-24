/**
 * app.js — Cliente interactivo para NLP Pipeline API
 * Contrato: /api/v1/clean | /pos | /ner | /visualize/dep | /vectorize
 */

const baseUrlInput = document.getElementById('baseUrl');
const statusDot    = document.getElementById('statusDot');
const statusTxt    = document.getElementById('statusTxt');

function getBaseUrl() {
  return baseUrlInput.value.trim().replace(/\/$/, '');
}

// ─── Health check ────────────────────────────────────────────────────────────
async function pingHealth() {
  try {
    const res = await fetch(`${getBaseUrl()}/`, { method: 'GET' });
    statusDot.className = res.ok ? 'dot ok' : 'dot bad';
    statusTxt.textContent = res.ok ? 'conectado' : `HTTP ${res.status}`;
  } catch {
    statusDot.className = 'dot bad';
    statusTxt.textContent = 'sin conexión';
  }
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────
function initTabs() {
  const tabs   = document.querySelectorAll('#tabs button');
  const routes = document.querySelectorAll('.route');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      routes.forEach(r => r.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`route-${btn.dataset.route}`)?.classList.add('active');
    });
  });
}

// ─── Lista dinámica de documentos ────────────────────────────────────────────
function setupDynamicList(listId, addBtnId, removeBtnId) {
  const list      = document.getElementById(listId);
  const addBtn    = document.getElementById(addBtnId);
  const removeBtn = document.getElementById(removeBtnId);

  addBtn?.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'nuevo documento…';
    list.appendChild(input);
  });

  removeBtn?.addEventListener('click', () => {
    if (list.children.length > 1) list.removeChild(list.lastElementChild);
  });
}

// ─── Envío genérico ───────────────────────────────────────────────────────────
async function sendRequest(endpoint) {
  const btn = document.querySelector(`.send[data-endpoint="${endpoint}"]`);
  const out = document.querySelector(`[data-out="${endpoint}"]`);

  btn.disabled = true;
  out.classList.remove('empty');
  out.textContent = 'esperando respuesta…';

  // Construir body según endpoint
  let url, body, expectHtml = false;

  if (endpoint === 'clean') {
    url  = `${getBaseUrl()}/api/v1/clean`;
    const raw = document.querySelector('#route-clean textarea[data-text]').value.trim();
    body = JSON.stringify({ text: raw });

  } else if (endpoint === 'pos') {
    url  = `${getBaseUrl()}/api/v1/pos`;
    const raw = document.querySelector('#route-pos textarea[data-text]').value.trim();
    body = JSON.stringify({ text: raw });

  } else if (endpoint === 'ner') {
    url  = `${getBaseUrl()}/api/v1/ner`;
    const raw = document.querySelector('#route-ner textarea[data-text]').value.trim();
    body = JSON.stringify({ text: raw });

  } else if (endpoint === 'dep') {
    url        = `${getBaseUrl()}/api/v1/visualize/dep`;
    expectHtml = true;
    const raw = document.querySelector('#route-dep textarea[data-text]').value.trim();
    body = JSON.stringify({ text: raw });

  } else if (endpoint === 'vectorize') {
    url  = `${getBaseUrl()}/api/v1/vectorize`;
    const docs = Array.from(document.querySelectorAll('#vectorizeList input'))
      .map(i => i.value.trim()).filter(Boolean);
    body = JSON.stringify({ documents: docs });
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if (expectHtml) {
      // /visualize/dep devuelve text/html con el SVG de displaCy
      const html = await res.text();
      const depBox = document.getElementById('dep-preview');
      if (depBox) depBox.innerHTML = html;
      out.textContent = res.ok
        ? `✓ HTML recibido (${html.length} bytes). SVG renderizado arriba.`
        : `HTTP ${res.status}\n${html}`;
    } else {
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
      if (!res.ok) {
        out.innerHTML = `<span class="err">HTTP ${res.status}</span>\n` + JSON.stringify(data, null, 2);
      }
      // Renderizar tabla de vectorización
      if (endpoint === 'vectorize' && res.ok) renderVectorTable(data);
    }
  } catch (e) {
    out.innerHTML = `<span class="err">Error de red: ${e.message}</span>\n\n¿Está uvicorn corriendo? ¿El puerto está abierto en el Security Group?`;
  } finally {
    btn.disabled = false;
  }
}

// ─── Tabla de vectorización ───────────────────────────────────────────────────
function renderVectorTable(data) {
  const container = document.getElementById('vector-table-container');
  if (!container) return;
  container.innerHTML = '';

  const { vocabulary, bag_of_words, tf_idf, one_hot } = data;
  if (!vocabulary || !vocabulary.length) return;

  function makeTable(title, rows) {
    const headers = vocabulary.map(v => `<th>${v}</th>`).join('');
    const bodyRows = rows.map((row, i) =>
      `<tr><td><strong>Doc ${i + 1}</strong></td>${row.map(v => `<td>${v}</td>`).join('')}</tr>`
    ).join('');
    return `
      <div style="margin-top:16px;">
        <div class="step-title">${title}</div>
        <div class="matrix-table-wrap">
          <table class="matrix-table">
            <thead><tr><th>Doc</th>${headers}</tr></thead>
            <tbody>${bodyRows}</tbody>
          </table>
        </div>
      </div>`;
  }

  container.innerHTML =
    makeTable('Bag of Words', bag_of_words) +
    makeTable('TF-IDF', tf_idf);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  baseUrlInput.addEventListener('change', pingHealth);
  pingHealth();

  setupDynamicList('vectorizeList', 'addDoc', 'removeDoc');

  document.querySelectorAll('.send').forEach(btn => {
    btn.addEventListener('click', () => sendRequest(btn.dataset.endpoint));
  });
});
