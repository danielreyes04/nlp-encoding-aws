/**
 * app.js — Cliente interactivo NLP Pipeline API
 * Corpus compartido: todas las pestañas leen de #sharedCorpus.
 */

const baseUrlInput = document.getElementById('baseUrl');
const statusDot    = document.getElementById('statusDot');
const statusTxt    = document.getElementById('statusTxt');

function getBaseUrl() {
  return baseUrlInput.value.trim().replace(/\/$/, '');
}

/** Lee los documentos del corpus compartido (filtra vacíos). */
function getCorpus() {
  return Array.from(document.querySelectorAll('#sharedCorpus input'))
    .map(i => i.value.trim())
    .filter(Boolean);
}

// ─── Health check ─────────────────────────────────────────────────────────────
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

// ─── Tabs ──────────────────────────────────────────────────────────────────────
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

// ─── Selector de documento para /dep ──────────────────────────────────────────
function refreshDepSelector() {
  const sel = document.getElementById('depDocIndex');
  if (!sel) return;
  const corpus = getCorpus();
  const prev = sel.value;
  sel.innerHTML = corpus
    .map((doc, i) => `<option value="${i}">${i + 1}. ${doc.substring(0, 50)}${doc.length > 50 ? '…' : ''}</option>`)
    .join('');
  // Restaurar selección si sigue siendo válida
  if (prev && parseInt(prev) < corpus.length) sel.value = prev;
}

// ─── Lista dinámica del corpus compartido ─────────────────────────────────────
function setupSharedCorpus() {
  const list      = document.getElementById('sharedCorpus');
  const addBtn    = document.getElementById('addDoc');
  const removeBtn = document.getElementById('removeDoc');

  const onChange = () => refreshDepSelector();

  // Observar cambios en los inputs existentes
  list.querySelectorAll('input').forEach(inp => inp.addEventListener('input', onChange));

  addBtn?.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'nuevo documento…';
    input.addEventListener('input', onChange);
    list.appendChild(input);
    refreshDepSelector();
  });

  removeBtn?.addEventListener('click', () => {
    if (list.children.length > 1) {
      list.removeChild(list.lastElementChild);
      refreshDepSelector();
    }
  });
}

// ─── Envío de solicitudes ─────────────────────────────────────────────────────
async function sendRequest(endpoint) {
  const btn = document.querySelector(`.send[data-endpoint="${endpoint}"]`);
  const out = document.querySelector(`[data-out="${endpoint}"]`);

  btn.disabled = true;
  out.classList.remove('empty');
  out.textContent = 'esperando respuesta…';

  const corpus = getCorpus();
  let url, body, expectHtml = false;

  if (endpoint === 'clean') {
    url  = `${getBaseUrl()}/api/v1/clean`;
    body = JSON.stringify({ text: corpus });

  } else if (endpoint === 'pos') {
    url  = `${getBaseUrl()}/api/v1/pos`;
    body = JSON.stringify({ text: corpus });

  } else if (endpoint === 'ner') {
    url  = `${getBaseUrl()}/api/v1/ner`;
    body = JSON.stringify({ text: corpus });

  } else if (endpoint === 'dep') {
    // /visualize/dep solo acepta un único string
    const idx = parseInt(document.getElementById('depDocIndex').value) || 0;
    const text = corpus[idx] ?? corpus[0];
    url        = `${getBaseUrl()}/api/v1/visualize/dep`;
    expectHtml = true;
    body       = JSON.stringify({ text });

  } else if (endpoint === 'vectorize') {
    url  = `${getBaseUrl()}/api/v1/vectorize`;
    body = JSON.stringify({ documents: corpus });
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if (expectHtml) {
      const html = await res.text();
      const depBox = document.getElementById('dep-preview');
      if (depBox) depBox.innerHTML = html;
      out.textContent = res.ok
        ? `✓ SVG renderizado (documento ${(parseInt(document.getElementById('depDocIndex').value) || 0) + 1}).`
        : `HTTP ${res.status}\n${html}`;
    } else {
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
      if (!res.ok) {
        out.innerHTML = `<span class="err">HTTP ${res.status}</span>\n` + JSON.stringify(data, null, 2);
      }
      if (endpoint === 'vectorize' && res.ok) renderVectorTable(data);
    }
  } catch (e) {
    out.innerHTML = `<span class="err">Error de red: ${e.message}</span>\n\n¿Está uvicorn corriendo? ¿El puerto 8000 está abierto en el Security Group?`;
  } finally {
    btn.disabled = false;
  }
}

// ─── Tabla de vectorización ───────────────────────────────────────────────────
function renderVectorTable(data) {
  const container = document.getElementById('vector-table-container');
  if (!container) return;
  container.innerHTML = '';

  const { vocabulary, bag_of_words, tf_idf } = data;
  if (!vocabulary?.length) return;

  function makeTable(title, rows) {
    const headers  = vocabulary.map(v => `<th>${v}</th>`).join('');
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

// ─── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  setupSharedCorpus();
  refreshDepSelector();

  baseUrlInput.addEventListener('change', pingHealth);
  pingHealth();

  document.querySelectorAll('.send').forEach(btn => {
    btn.addEventListener('click', () => sendRequest(btn.dataset.endpoint));
  });
});
