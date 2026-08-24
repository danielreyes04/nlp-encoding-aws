/**
 * app.js — NLP Pipeline cliente (EC2 + Lambda en paralelo)
 *
 * Flujo:
 *  1. Usuario edita corpus y pulsa "Procesar corpus".
 *  2. Se llaman los 5 endpoints en EC2 y Lambda simultáneamente con
 *     Promise.allSettled — si uno falla el otro sigue mostrándose.
 *  3. Resultados se guardan en state.ec2 y state.lambda.
 *  4. Las pestañas muestran ambas respuestas lado a lado.
 *     Cambiar de doc-tab es instantáneo (sin fetch extra).
 */

// ─── Estado ───────────────────────────────────────────────────────────────────
const state = {
  corpus: [],
  ec2:    { clean: null, pos: null, ner: null, dep: [], vectorize: null },
  lambda: { clean: null, pos: null, ner: null, dep: [], vectorize: null },
};

// ─── DOM helpers ──────────────────────────────────────────────────────────────
function getEc2Url()    { return document.getElementById('ec2Url').value.trim().replace(/\/$/, ''); }
function getLambdaUrl() { return document.getElementById('lambdaUrl').value.trim().replace(/\/$/, ''); }
function getCorpus() {
  return Array.from(document.querySelectorAll('#sharedCorpus input'))
    .map(i => i.value.trim()).filter(Boolean);
}

// ─── Health ───────────────────────────────────────────────────────────────────
async function pingHealth() {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusTxt');
  const [r1, r2] = await Promise.allSettled([
    fetch(`${getEc2Url()}/`).then(r => r.ok),
    fetch(`${getLambdaUrl()}/`).then(r => r.ok),
  ]);
  const okEc2    = r1.status === 'fulfilled' && r1.value;
  const okLambda = r2.status === 'fulfilled' && r2.value;
  if (okEc2 && okLambda)      { dot.className = 'dot ok';  txt.textContent = 'EC2 ✓  Lambda ✓'; }
  else if (okEc2)             { dot.className = 'dot bad'; txt.textContent = 'EC2 ✓  Lambda ✗'; }
  else if (okLambda)          { dot.className = 'dot bad'; txt.textContent = 'EC2 ✗  Lambda ✓'; }
  else                        { dot.className = 'dot bad'; txt.textContent = 'sin conexión'; }
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

// ─── Doc-tabs ─────────────────────────────────────────────────────────────────
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
    btns[idx]?.classList.add('active');
    onSelect(idx);
  };
}

// ─── Renderizadores ───────────────────────────────────────────────────────────
function renderClean() {
  const n = state.corpus.length;
  const select = buildDocTabs(document.getElementById('clean-doc-tabs'), n, (i) => {
    const ev = state.ec2.clean?.cleaned_text?.[i] ?? null;
    const lv = state.lambda.clean?.cleaned_text?.[i] ?? null;
    setOut('clean-out-ec2',    ev !== null ? JSON.stringify({ doc: i+1, cleaned_text: ev }, null, 2) : '(sin respuesta)');
    setOut('clean-out-lambda', lv !== null ? JSON.stringify({ doc: i+1, cleaned_text: lv }, null, 2) : '(sin respuesta)');
  });
  select(0);
}

function renderPos() {
  const n = state.corpus.length;
  const select = buildDocTabs(document.getElementById('pos-doc-tabs'), n, (i) => {
    const ev = state.ec2.pos?.results?.[i] ?? null;
    const lv = state.lambda.pos?.results?.[i] ?? null;
    setOut('pos-out-ec2',    ev ? JSON.stringify({ doc: i+1, tokens: ev }, null, 2) : '(sin respuesta)');
    setOut('pos-out-lambda', lv ? JSON.stringify({ doc: i+1, tokens: lv }, null, 2) : '(sin respuesta)');
  });
  select(0);
}

function renderNer() {
  const n = state.corpus.length;
  const select = buildDocTabs(document.getElementById('ner-doc-tabs'), n, (i) => {
    const ev = state.ec2.ner?.results?.[i] ?? null;
    const lv = state.lambda.ner?.results?.[i] ?? null;
    setOut('ner-out-ec2',    ev !== null ? JSON.stringify({ doc: i+1, entities: ev }, null, 2) : '(sin respuesta)');
    setOut('ner-out-lambda', lv !== null ? JSON.stringify({ doc: i+1, entities: lv }, null, 2) : '(sin respuesta)');
  });
  select(0);
}

function renderDep() {
  const n = state.corpus.length;
  const select = buildDocTabs(document.getElementById('dep-doc-tabs'), n, (i) => {
    document.getElementById('dep-preview-ec2').innerHTML =
      state.ec2.dep[i]    ?? '<div class="empty-hint">sin respuesta de EC2</div>';
    document.getElementById('dep-preview-lambda').innerHTML =
      state.lambda.dep[i] ?? '<div class="empty-hint">sin respuesta de Lambda</div>';
  });
  select(0);
}

function renderVectorize() {
  const ev = state.ec2.vectorize;
  const lv = state.lambda.vectorize;
  setOut('vectorize-out-ec2',    ev ? JSON.stringify(ev, null, 2) : '(sin respuesta)');
  setOut('vectorize-out-lambda', lv ? JSON.stringify(lv, null, 2) : '(sin respuesta)');
  buildVectorTable('vector-table-ec2',    ev);
  buildVectorTable('vector-table-lambda', lv);
}

function buildVectorTable(containerId, data) {
  const el = document.getElementById(containerId);
  if (!data?.vocabulary?.length) {
    el.innerHTML = '<div class="empty-hint">sin datos</div>';
    return;
  }
  const { vocabulary, bag_of_words, tf_idf } = data;
  const makeTable = (title, rows) => {
    const ths = vocabulary.map(v => `<th>${v}</th>`).join('');
    const trs = rows.map((row, i) =>
      `<tr><td><strong>D${i+1}</strong></td>${row.map(v => `<td>${v}</td>`).join('')}</tr>`
    ).join('');
    return `<div style="margin-top:12px;">
      <div class="step-title">${title}</div>
      <div class="matrix-table-wrap">
        <table class="matrix-table">
          <thead><tr><th></th>${ths}</tr></thead>
          <tbody>${trs}</tbody>
        </table>
      </div></div>`;
  };
  el.innerHTML = makeTable('Bag of Words', bag_of_words) + makeTable('TF-IDF', tf_idf);
}

function setOut(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('empty');
  el.textContent = text;
}

// ─── Fetch helpers (nunca lanzan excepción) ───────────────────────────────────
async function safeJsonPost(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.ok ? r.json() : null;
  } catch { return null; }
}

async function safeHtmlPost(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.ok ? r.text() : null;
  } catch { return null; }
}

// ─── Procesamiento principal ──────────────────────────────────────────────────
async function processCorpus() {
  const corpus = getCorpus();
  if (!corpus.length) return;

  const btn    = document.getElementById('processBtn');
  const status = document.getElementById('processStatus');
  btn.disabled = true;
  status.textContent = 'procesando…';
  status.className   = 'process-status running';

  state.corpus = corpus;
  const ec2    = getEc2Url();
  const lam    = getLambdaUrl();

  // Todas las peticiones en paralelo. safe* nunca lanzan excepción.
  const [
    ec2Clean, ec2Pos, ec2Ner, ec2Vec,
    lClean,   lPos,   lNer,   lVec,
    ...depAll
  ] = await Promise.all([
    // EC2 — batch
    safeJsonPost(`${ec2}/api/v1/clean`,     { text: corpus }),
    safeJsonPost(`${ec2}/api/v1/pos`,       { text: corpus }),
    safeJsonPost(`${ec2}/api/v1/ner`,       { text: corpus }),
    safeJsonPost(`${ec2}/api/v1/vectorize`, { documents: corpus }),
    // Lambda — batch
    safeJsonPost(`${lam}/api/v1/clean`,     { text: corpus }),
    safeJsonPost(`${lam}/api/v1/pos`,       { text: corpus }),
    safeJsonPost(`${lam}/api/v1/ner`,       { text: corpus }),
    safeJsonPost(`${lam}/api/v1/vectorize`, { documents: corpus }),
    // dep: N peticiones EC2 + N peticiones Lambda
    ...corpus.map(text => safeHtmlPost(`${ec2}/api/v1/visualize/dep`, { text })),
    ...corpus.map(text => safeHtmlPost(`${lam}/api/v1/visualize/dep`, { text })),
  ]);

  state.ec2.clean     = ec2Clean;
  state.ec2.pos       = ec2Pos;
  state.ec2.ner       = ec2Ner;
  state.ec2.vectorize = ec2Vec;
  state.ec2.dep       = depAll.slice(0, corpus.length);

  state.lambda.clean     = lClean;
  state.lambda.pos       = lPos;
  state.lambda.ner       = lNer;
  state.lambda.vectorize = lVec;
  state.lambda.dep       = depAll.slice(corpus.length);

  renderClean();
  renderPos();
  renderNer();
  renderDep();
  renderVectorize();
  applyEnvFilter(); // aplicar el filtro activo sobre los datos recién cargados

  const okEc2    = !!(ec2Clean || ec2Pos || ec2Ner || ec2Vec);
  const okLambda = !!(lClean   || lPos   || lNer   || lVec);
  const label    = okEc2 && okLambda ? 'EC2 ✓  Lambda ✓'
                 : okEc2             ? 'EC2 ✓  Lambda ✗ (timeout o error)'
                 : okLambda          ? 'EC2 ✗  Lambda ✓'
                 :                    'EC2 ✗  Lambda ✗';

  status.textContent = `✓ ${corpus.length} doc${corpus.length > 1 ? 's' : ''} — ${label}`;
  status.className   = (okEc2 && okLambda) ? 'process-status ok' : 'process-status running';
  btn.disabled = false;
}

// ─── Selector Mostrar (EC2 / Lambda / Ambos) ─────────────────────────────────
function applyEnvFilter() {
  const val = document.getElementById('activeEnv').value;
  document.querySelectorAll('.dual-col').forEach(col => {
    const isEc2    = col.querySelector('.ec2-label') !== null;
    const isLambda = col.querySelector('.lambda-label') !== null;
    if (val === 'both') {
      col.style.display = '';
    } else if (val === 'ec2' && isEc2) {
      col.style.display = '';
    } else if (val === 'lambda' && isLambda) {
      col.style.display = '';
    } else {
      col.style.display = 'none';
    }
  });
  // Cuando solo se muestra uno, ocupa el ancho completo
  document.querySelectorAll('.dual-out').forEach(row => {
    row.style.gridTemplateColumns = val === 'both' ? '' : '1fr';
  });
}

// ─── Corpus dinámico ──────────────────────────────────────────────────────────
function setupSharedCorpus() {
  const list = document.getElementById('sharedCorpus');
  document.getElementById('addDoc')?.addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'text'; inp.placeholder = 'nuevo documento…';
    list.appendChild(inp);
  });
  document.getElementById('removeDoc')?.addEventListener('click', () => {
    if (list.children.length > 1) list.removeChild(list.lastElementChild);
  });
}

// ─── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  setupSharedCorpus();
  pingHealth();
  document.getElementById('ec2Url').addEventListener('change', pingHealth);
  document.getElementById('lambdaUrl').addEventListener('change', pingHealth);
  document.getElementById('activeEnv').addEventListener('change', applyEnvFilter);
  document.getElementById('processBtn').addEventListener('click', processCorpus);
});
