/* Nepal GovAgent — frontend logic */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const form     = $('ask-form');
  const input    = $('ask-input');
  const button   = $('ask-btn');
  const status   = $('status');
  const answer   = $('answer');
  const body     = $('answer-body');
  const sources  = $('sources');
  const conf     = $('confidence');
  const elapsed  = $('elapsed');

  // ---- helpers -----------------------------------------------------------

  function showStatus(text, kind) {
    status.hidden = false;
    status.textContent = text;
    status.className = 'status' + (kind ? ' status--' + kind : '');
  }
  function hideStatus()  { status.hidden = true; status.textContent = ''; status.className = 'status'; }
  function hideAnswer()  { answer.hidden = true; }
  function showAnswer()  { answer.hidden = false; }

  function escapeHTML(s) {
    return String(s ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function renderSources(list) {
    sources.innerHTML = '';
    if (!list || !list.length) {
      const li = document.createElement('li');
      li.className = 'source';
      li.innerHTML = '<p class="source__doc">No sources returned.</p>';
      sources.appendChild(li);
      return;
    }
    for (const s of list) {
      const li = document.createElement('li');
      const heading = s.heading ? ` · ${escapeHTML(s.heading)}` : '';
      li.innerHTML = `
        <p class="source__doc">${escapeHTML(s.doc)}</p>
        <p class="source__meta">Page ${escapeHTML(String(s.page))}${heading}</p>
        ${s.excerpt ? `<p class="source__excerpt">${escapeHTML(s.excerpt)}</p>` : ''}
      `;
      sources.appendChild(li);
    }
  }

  function setConfidence(level) {
    const known = ['high', 'medium', 'low'];
    const k = known.includes(level) ? level : 'low';
    conf.textContent = k;
    conf.className = 'confidence confidence--' + k;
  }

  function setBusy(busy) {
    button.disabled = busy;
    input.disabled = busy;
  }

  // ---- ask ---------------------------------------------------------------

  async function ask(query) {
    hideAnswer();
    showStatus('स्रोत खोज्दै… · Retrieving sources…');
    setBusy(true);

    const t0 = performance.now();
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k: 5 }),
      });

      if (res.status === 503) {
        await warmThenRetry(query);
        return;
      }
      if (!res.ok) {
        const detail = await safeJsonDetail(res);
        showStatus(`Request failed (${res.status}): ${detail}`, 'error');
        return;
      }

      const data = await res.json();
      hideStatus();
      const text = data.answer || '(no answer assembled)';
      body.textContent = '';
      text.split(/\n{2,}/).forEach((para) => {
        const p = document.createElement('p');
        p.textContent = para.trim();
        if (p.textContent) body.appendChild(p);
      });
      if (!body.childNodes.length) {
        const p = document.createElement('p');
        p.textContent = text.trim();
        body.appendChild(p);
      }
      setConfidence(data.confidence);
      elapsed.textContent = `${data.elapsed_ms} ms`;
      renderSources(data.sources);
      showAnswer();
    } catch (err) {
      showStatus(`Network error: ${err.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }

  async function safeJsonDetail(res) {
    try { const j = await res.json(); return j.detail || JSON.stringify(j); }
    catch { return res.statusText; }
  }

  // ---- warm-up flow ------------------------------------------------------

  async function warmThenRetry(query) {
    showStatus(
      'सेवा सुरु हुँदैछ — पहिलो अनुरोधमा ~३० सेकेन्ड लाग्न सक्छ (मोडेल लोड)। ' +
      'Waking up — first request after idle may take ~30s while embeddings load.',
      'warming'
    );
    // Poll /api/health until ready, then retry once.
    const deadline = Date.now() + 90_000;
    while (Date.now() < deadline) {
      await sleep(2500);
      try {
        const r = await fetch('/api/health');
        const j = await r.json();
        if (j.ready) {
          showStatus('तयार। स्रोत खोज्दै… · Ready. Retrieving…');
          await sleep(400);
          await ask(query);
          return;
        }
        if (j.error) {
          showStatus(`Service failed to start: ${j.error}`, 'error');
          setBusy(false);
          return;
        }
      } catch (_) { /* keep polling */ }
    }
    showStatus('Service is taking longer than expected. Try again in a minute.', 'error');
    setBusy(false);
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---- stats -------------------------------------------------------------

  async function loadStats() {
    try {
      const r = await fetch('/api/stats');
      if (!r.ok) return;
      const j = await r.json();
      if (!j || !j.ready) return;
      const docs = $('stat-docs');
      const blocks = $('stat-blocks');
      const offline = $('stat-offline');
      if (docs)    { docs.textContent = String(j.documents ?? '—'); docs.classList.add('is-number'); }
      if (blocks)  { blocks.textContent = (j.blocks ?? '—').toLocaleString?.() ?? String(j.blocks); blocks.classList.add('is-number'); }
      if (offline) { offline.textContent = j.offline ? 'offline' : 'online'; }
    } catch (_) { /* silent */ }
  }

  // ---- wiring ------------------------------------------------------------

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) ask(q);
  });

  // Enter submits; Shift+Enter inserts newline
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll('.example-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      input.value = btn.dataset.q || '';
      input.focus();
      form.requestSubmit();
    });
  });

  loadStats();
  // Try once more after a delay in case backend was still warming on load
  setTimeout(loadStats, 5000);
})();
