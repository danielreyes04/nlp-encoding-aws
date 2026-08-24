/**
 * app.js — NLP Pipeline cliente
 *
 * Flujo:
 *  1. Usuario edita el corpus compartido.
 *  2. Pulsa "Procesar corpus" → se llaman todos los endpoints en paralelo.
 *  3. Los resultados se guardan en `state`.
 *  4. Las pestañas renderizan desde `state` con mini-tabs por documento.
 *     Cambiar de pestaña o de documento no hace ninguna petición extra.
 */

// ─── Estado global ────────────────────────────────────────────────────────────
const state = {
  corpus:    [],   // strings originales
  clean:     null, // { cleaned_text: [...] }
  pos:       null, // { results: [[...], ...] }
  ner:       null, // { results: [[...], ...] }
  dep:       [],   // array de strings HTML, uno por doc
  vectorize: null, // { vocabulary, bag_of_words, one_hot, tf_idf }
};

// ─── Helpers DOM ──────────────────────────────────────────────────────────────
const baseUrlInput = document.getElementById('baseUrl');
const statusDot    = document.getElementById('statusDot');
const statusTxt    = document.getElementById('statusTxt');

function getBaseUrl() {
  return baseUrlInput.value.trim().replace(/\/$/, '');
}

function getCorpus() {
  return Array.from(document.querySelectorAll('#sharedCorpus input'))
    .map(i => i.value.trim())
    .filter(Boolean);
}

// ─── Health ───────────────────────────────────────────────────────────────────
async function pingHealth() {
  try {
    const res = await fetch(`${getBaseUrl()}/`);
    statusDot.className = res.ok ? 'dot ok' : 'dot bad';
    statusTxt.textContent = res.ok ? 'conectado' : `HTTP ${res.status}`;
  } catch {
    statusDot.className = 'dot bad';
    statusTxt.textContent = 'sin conexión';
  }
}

// ─── Tabs de sección ──────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('#tabs button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.route').forEach(r => r.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`route-${btn.dataset.route}`)?.classList.add('active');
    });
  });
}

// ─── Mini-tabs por documento ──────────────────────────────────────────────────
/**
 * Crea mini-tabs numerados (Doc 1, Doc 2 …) dentro de `containerEl`.
 * Cuando el usuario pulsa uno, llama onSelect(index).
 * Retorna la función selectTab(index) para poder activar el primero externamente.
 */
function buildDocTabs(containerEl, count, onSelect) {
  containerEl.innerHTML = '';
  const btns = [];

  for (let i = 0; i < count; i++) {
    const btn = document.createElement('button');
    btn.className = 'doc-tab';
    btn.textContent = `Doc ${i + 1}`;
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      onSelect(i);
    });
    containerEl.appendChild(btn);
    btns.push(btn);
  }

  return (idx) => {
    btns.forEach(b => b.classList.remove('active'));
    if (btns[idx]) btns[idx].classList.add('active');
    onSelect(idx);
  };
}

// ─── Renderizadores por sección ───────────────────────────────────────────────

function renderClean() {
  const tabsEl = document.getElementById('clean-doc-tabs');
  const outEl  = document.getElementById('clean-out');
  if (!state.clean) return;

  const items = state.clean.cleaned_text;
  outEl.classList.remove('empty');

  const select = buildDocTabs(tabsEl, items.length, (i) => {
    outEl.textContent = JSON.stringify({ doc: i + 1, cleaned_text: items[i] }, null, 2);
  });
  select(0);
}

function renderPos() {
  const tabsEl = document.getElementById('pos-doc-tabs');
  const outEl  = document.getElementById('pos-out');
  if (!state.pos) return;

  const results = state.pos.results;
  outEl.classList.remove('empty');

  const select = buildDocTabs(tabsEl, results.length, (i) => {
    outEl.textContent = JSON.stringify({ doc: i + 1, tokens: results[i] }, null, 2);
  });
  select(0);
}

function renderNer() {
  const tabsEl = document.getElementById('ner-doc-tabs');
  const outEl  = document.getElementById('ner-out');
  if (!state.ner) return;

  const results = state.ner.results;
  outEl.classList.remove('empty');

  const select = buildDocTabs(tabsEl, results.length, (i) => {
    outEl.textContent = JSON.stringify({ doc: i + 1, entities: results[i] }, null, 2);
  });
  select(0);
}

function renderDep() {
  const tabsEl  = document.getElementById('dep-doc-tabs');
  const preview = document.getElementById('dep-preview');
  if (!state.dep.length) return;

  preview.innerHTML = '';

  // Muestra el SVG del documento seleccionado
  const select = buildDocTabs(tabsEl, state.dep.length, (i) => {
    preview.innerHTML = state.dep[i];  // HTML completo con el SVG
  });
  select(0);
}

function renderVectorize() {
  const container = document.getElementById('vector-table-container');
  const outEl     = document.getElementById('vectorize-out');
  if (!state.vectorize) return;

  outEl.classList.remove('empty');
  outEl.textContent = JSON.stringify(state.vectorize, null, 2);

  const { vocabulary, bag_of_words, tf_idf } = state.vectorize;
  if (!vocabulary?.length) return;

  function makeTable(title, rows) {
    const ths  = vocabulary.map(v => `<th>${v}</th>`).join('');
    const trs  = rows.map((row, i) =>
      `<tr><td><strong>Doc ${i + 1}</strong></td>${row.map(v => `<td>${v}</td>`).join('')}</tr>`
    ).join('');
    return `
      <div style="margin-top:16px;">
        <div class="step-title">${title}</div>
        <div class="matrix-table-wrap">
          <table class="matrix-table">
            <thead><tr><th>Doc</th>${ths}</tr></thead>
            <tbody>${trs}</tbody>
          </table>
        </div>
      </div>`;
  }

  container.innerHTML =
    makeTable('Bag of Words', bag_of_words) +
    makeTable('TF-IDF', tf_idf);
}

// ─── Procesamiento del corpus ─────────────────────────────────────────────────
async function processCorpus() {
  const corpus = getCorpus();
  if (!corpus.length) return;

  const btn    = document.getElementById('processBtn');
  const status = document.getElementById('processStatus');

  btn.disabled = true;
  status.textContent = 'procesando…';
  status.className   = 'process-status running';

  const base = getBaseUrl();
  state.corpus = corpus;

  // Llamadas a clean / pos / ner / vectorize en paralelo.
  // dep necesita una petición por documento (endpoint no admite batch).
  const depRequests = corpus.map(text =>
    fetch(`${base}/api/v1/visualize/dep`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }).then(r => r.text())
  );

  try {
    const [cleanRes, posRes, nerRes, vectorizeRes, ...depResolved] = await Promise.all([
      fetch(`${base}/api/v1/clean`,     { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: corpus }) }).then(r => r.json()),
      fetch(`${base}/api/v1/pos`,       { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: corpus }) }).then(r => r.json()),
      fetch(`${base}/api/v1/ner`,       { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: corpus }) }).then(r => r.json()),
      fetch(`${base}/api/v1/vectorize`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ documents: corpus }) }).then(r => r.json()),
      ...depRequests,
    ]);

    state.clean     = cleanRes;
    state.pos       = posRes;
    state.ner       = nerRes;
    state.vectorize = vectorizeRes;
    state.dep       = depResolved;

    // Renderizar todas las vistas
    renderClean();
    renderPos();
    renderNer();
    renderDep();
    renderVectorize();

    status.textContent = `✓ listo — ${corpus.length} documento${corpus.length > 1 ? 's' : ''} procesado${corpus.length > 1 ? 's' : ''}`;
    status.className   = 'process-status ok';
  } catch (e) {
    status.textContent = `✗ error: ${e.message}`;
    status.className   = 'process-status error';
  } finally {
    btn.disabled = false;
  }
}

// ─── Corpus dinámico ──────────────────────────────────────────────────────────
function setupSharedCorpus() {
  const list      = document.getElementById('sharedCorpus');
  const addBtn    = document.getElementById('addDoc');
  const removeBtn = document.getElementById('removeDoc');

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

// ─── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  setupSharedCorpus();

  baseUrlInput.addEventListener('change', pingHealth);
  pingHealth();

  document.getElementById('processBtn').addEventListener('click', processCorpus);
});
