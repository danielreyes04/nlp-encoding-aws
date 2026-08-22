/**
 * app.js - Lógica del cliente interactivo para NLP Pipeline API
 */

const baseUrlInput = document.getElementById('baseUrl');
const statusDot = document.getElementById('statusDot');
const statusTxt = document.getElementById('statusTxt');

function getBaseUrl() {
  return baseUrlInput.value.trim().replace(/\/$/, '');
}

// ---------------------------------------------------------------------
// 1. Verificación de conexión (Health Check)
// ---------------------------------------------------------------------
async function pingHealth() {
  try {
    const res = await fetch(`${getBaseUrl()}/`, { method: 'GET' });
    if (res.ok) {
      statusDot.className = 'dot ok';
      statusTxt.textContent = 'conectado';
    } else {
      statusDot.className = 'dot bad';
      statusTxt.textContent = `HTTP ${res.status}`;
    }
  } catch (e) {
    statusDot.className = 'dot bad';
    statusTxt.textContent = 'sin conexión';
  }
}

// ---------------------------------------------------------------------
// 2. Control de Pestañas (Tabs)
// ---------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('#tabs button');
  const routes = document.querySelectorAll('.route');

  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      routes.forEach(r => r.classList.remove('active'));

      btn.classList.add('active');
      const targetRoute = document.getElementById(`route-${btn.dataset.route}`);
      if (targetRoute) targetRoute.classList.add('active');
    });
  });
}

// ---------------------------------------------------------------------
// 3. Manejo de listas dinámicas para Corpus (/encoding y /pipeline)
// ---------------------------------------------------------------------
function setupDynamicList(listId, addBtnId, removeBtnId) {
  const list = document.getElementById(listId);
  const addBtn = document.getElementById(addBtnId);
  const removeBtn = document.getElementById(removeBtnId);

  if (addBtn && list) {
    addBtn.addEventListener('click', () => {
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'nuevo documento…';
      list.appendChild(input);
    });
  }

  if (removeBtn && list) {
    removeBtn.addEventListener('click', () => {
      if (list.children.length > 1) {
        list.removeChild(list.lastElementChild);
      }
    });
  }
}

// ---------------------------------------------------------------------
// 4. Renderizadores visuales (Árboles displaCy y Paso a Paso)
// ---------------------------------------------------------------------
function renderTrees(treeBox, arboles) {
  if (!treeBox || !Array.isArray(arboles) || arboles.length === 0) return;

  arboles.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'tree-card';
    card.innerHTML = `
      <div class="tree-card-title">Árbol de dependencias — Oración ${i + 1}: "${item.oracion}"</div>
      <div>${item.svg}</div>
    `;
    treeBox.appendChild(card);
  });
}

function renderPipelineStepByStep(container, data) {
  if (!container || !data.paso_a_paso_documentos) return;

  // 1. Resumen
  const summary = document.createElement('div');
  summary.className = 'doc-step-card';
  summary.innerHTML = `
    <div class="doc-step-header">
      <span>RESUMEN DEL CORPUS</span>
      <span class="step-badge">Método: ${data.resumen.metodo_codificacion.toUpperCase()}</span>
    </div>
    <div style="font-size:12px; color:var(--dim);">
      Total de documentos procesados: <strong style="color:var(--ink);">${data.resumen.total_documentos}</strong> | 
      Tamaño del vocabulario global: <strong style="color:var(--ink);">${data.resumen.tamano_vocabulario} lemas</strong>
    </div>
  `;
  container.appendChild(summary);

  // 2. Desglose por documento
  data.paso_a_paso_documentos.forEach(doc => {
    const card = document.createElement('div');
    card.className = 'doc-step-card';

    const tokensHtml = doc.paso_1_processed.tokens_limpios.map(t =>
      `<span class="token-chip">${t.texto_original} ➔ <em>${t.lema}</em> <span class="chip-tag">${t.pos}</span></span>`
    ).join('') || '<span style="color:var(--dim); font-size:12px;">Sin tokens procesados</span>';

    let treesHtml = '<span style="color:var(--dim); font-size:12px;">Sin árboles generados</span>';
    if (doc.paso_2_dependency && Array.isArray(doc.paso_2_dependency.arboles) && doc.paso_2_dependency.arboles.length > 0) {
      treesHtml = doc.paso_2_dependency.arboles.map((arb, idx) => `
        <div class="tree-card" style="margin-top:8px;">
          <div class="tree-card-title">Oración ${idx + 1}: "${arb.oracion}"</div>
          <div>${arb.svg}</div>
        </div>
      `).join('');
    }

    let entsHtml = '<span style="color:var(--dim); font-size:12px;">No se detectaron entidades nombradas.</span>';
    if (doc.paso_3_ner && doc.paso_3_ner.entidades && doc.paso_3_ner.entidades.length > 0) {
      entsHtml = doc.paso_3_ner.entidades.map(e =>
        `<span class="token-chip" style="border-color:var(--warn);">${e.texto} <span class="chip-tag" style="color:var(--warn);">${e.etiqueta}</span></span>`
      ).join('');
    }

    card.innerHTML = `
      <div class="doc-step-header">
        <span>DOCUMENTO ${doc.id}: "${doc.documento_original}"</span>
      </div>
      <div class="step-section">
        <div class="step-title">Paso 1: Procesamiento (Limpieza, Lematización y POS)</div>
        <div class="token-chips">${tokensHtml}</div>
        <div style="margin-top:8px; font-size:12px; color:var(--dim);">Lemas limpios: <code>[${doc.paso_1_processed.lemas.join(', ')}]</code></div>
      </div>
      <div class="step-section">
        <div class="step-title">Paso 2: Dependencias Sintácticas y Árboles (spaCy displaCy)</div>
        <div>${treesHtml}</div>
      </div>
      <div class="step-section">
        <div class="step-title">Paso 3: Reconocimiento de Entidades Nombradas (NER)</div>
        <div class="token-chips">${entsHtml}</div>
      </div>
    `;
    container.appendChild(card);
  });

  // 3. Paso 5: Matriz de codificación
  if (data.paso_5_encoding_corpus) {
    const enc = data.paso_5_encoding_corpus;
    const encCard = document.createElement('div');
    encCard.className = 'doc-step-card';

    const tableHeaders = enc.vocabulario.map(v => `<th>${v}</th>`).join('');
    const tableRows = enc.documentos.map((d, i) => `
      <tr>
        <td><strong>Doc ${i + 1}</strong>: <em>${d.documento_original}</em></td>
        ${d.vector.map(val => `<td>${val}</td>`).join('')}
      </tr>
    `).join('');

    encCard.innerHTML = `
      <div class="doc-step-header">
        <span>PASO 5: CODIFICACIÓN DEL CORPUS COMPLETO (${enc.metodo.toUpperCase()})</span>
      </div>
      <div style="font-size:12px; color:var(--dim); margin-bottom:8px;">
        Vocabulario construido: <code>[${enc.vocabulario.join(', ')}]</code>
      </div>
      <div class="matrix-table-wrap">
        <table class="matrix-table">
          <thead>
            <tr>
              <th>Documento</th>
              ${tableHeaders}
            </tr>
          </thead>
          <tbody>
            ${tableRows}
          </tbody>
        </table>
      </div>
    `;
    container.appendChild(encCard);
  }
}

// ---------------------------------------------------------------------
// 5. Envío de solicitudes a los endpoints HTTP
// ---------------------------------------------------------------------
async function sendRequest(endpoint) {
  const btn = document.querySelector(`.send[data-endpoint="${endpoint}"]`);
  const out = document.querySelector(`[data-out="${endpoint}"]`);
  const treeBox = document.getElementById(`trees-${endpoint}`);
  const pipelineVisual = document.getElementById('pipeline-visual-container');

  if (treeBox) treeBox.innerHTML = '';
  if (pipelineVisual && endpoint === 'pipeline') pipelineVisual.innerHTML = '';

  btn.disabled = true;
  out.classList.remove('empty');
  out.textContent = 'esperando respuesta…';

  let body;
  if (endpoint === 'encoding' || endpoint === 'pipeline') {
    const listId = endpoint === 'pipeline' ? 'pipelineCorpusList' : 'corpusList';
    const methodId = endpoint === 'pipeline' ? 'pipelineMethod' : 'encodingMethod';
    const corpus = Array.from(document.querySelectorAll(`#${listId} input`))
      .map(i => i.value.trim())
      .filter(Boolean);
    const method = document.getElementById(methodId).value;
    body = JSON.stringify({ corpus, method });
  } else {
    const text = document.querySelector(`#route-${endpoint} textarea[data-text]`).value.trim();
    body = JSON.stringify({ text });
  }

  try {
    const res = await fetch(`${getBaseUrl()}/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);

    // Árboles para /dependency y /full
    if (treeBox) {
      const arboles = data.arboles || (data.dependencias && data.dependencias.arboles);
      renderTrees(treeBox, arboles);
    }

    // Paso a paso para /pipeline
    if (pipelineVisual && endpoint === 'pipeline') {
      renderPipelineStepByStep(pipelineVisual, data);
    }

    if (!res.ok) {
      out.innerHTML = `<span class="err">HTTP ${res.status}</span>\n` + JSON.stringify(data, null, 2);
    }
  } catch (e) {
    out.innerHTML = `<span class="err">Error de red: ${e.message}</span>\n\n¿Corriste uvicorn? ¿El puerto está abierto en el Security Group?`;
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------
// 6. Inicialización de la Aplicación
// ---------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  baseUrlInput.addEventListener('change', pingHealth);
  pingHealth();

  setupDynamicList('corpusList', 'addDoc', 'removeDoc');
  setupDynamicList('pipelineCorpusList', 'pipelineAddDoc', 'pipelineRemoveDoc');

  document.querySelectorAll('.send').forEach(btn => {
    btn.addEventListener('click', () => sendRequest(btn.dataset.endpoint));
  });
});
