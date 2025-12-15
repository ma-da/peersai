// app.js — Seeds of Truth UI logic
'use strict';

/* =========================================================
   0) TUNABLES / CONSTANTS
   ========================================================= */
const CFG = {
  // LocalStorage keys
  LS_THEME: 'sot-theme',
  LS_SIDEBAR_COLLAPSED: 'sot-sidebar-collapsed',
  LS_TOOLS: 'sot-tools',
  LS_SAVED_CONVOS: 'sot-saved-conversations-v1',
  LS_FEEDBACK: 'sot-feedback-v1',

  // Limits
  MAX_CONTEXT_TURNS: 5,
  MAX_CONVO_TURNS: 50,
  MAX_SAVED_CONVOS: 25,
  MAX_FEEDBACK_ITEMS: 200,
  MAX_REFS: 10,

  // UI behavior
  DEFAULT_THEME: 'light',      // 'light' | 'dark'
  DEFAULT_HISTORY_TURNS: 2,    // 0..5
  DEFAULT_MODE: 'chat',        // 'search' | 'chat' | 'ab'
  TEXTAREA_MAX_HEIGHT: 140,    // px
  STUB_DELAY_MS: 500,          // ms

  // Flask endpoints (adjust to match app.py)
  API: {
    CHAT: '/api/chat',
    FEEDBACK: '/api/feedback',
    STATUS: '/api/status',
    QUEUE: '/api/queue'
  }
};

/* =========================================================
   1) DOM LOOKUPS (set in init)
   ========================================================= */
const els = {};   // populated in initDom()

/* =========================================================
   2) STATE
   ========================================================= */
const toolState = {
  historyTurns: CFG.DEFAULT_HISTORY_TURNS,   // 0-5
  mode: CFG.DEFAULT_MODE                      // 'search' | 'chat' | 'ab'
};

// client-side turns: { user: string, assistant: string }
const convoTurns = [];

let botMsgCounter = 0;

// Feedback modal state
let feedbackTarget = null; // { type, id, snippet }

/* =========================================================
   3) UTILITIES
   ========================================================= */
function clampInt(n, min, max, fallback) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return fallback;
  return Math.max(min, Math.min(max, x));
}

function clamp1to10(n) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return null;
  return Math.max(1, Math.min(10, x));
}

function safeJsonParse(str, fallback) {
  try { return JSON.parse(str); } catch (_) { return fallback; }
}

function nowId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function formatDuration(totalSeconds) {
  const s = Math.max(0, totalSeconds | 0);
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return `${r}s`;
  return `${m}m ${r}s`;
}

/* =========================================================
   4) MODAL (custom alert/confirm)
   ========================================================= */
let modalResolve = null;

function openModal({ title, message, buttons }) {
  els.modalTitle.textContent = title || 'Notice';
  els.modalMessage.textContent = message || '';
  els.modalActions.innerHTML = '';

  buttons.forEach((b) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `modal-btn${b.variant ? ' ' + b.variant : ''}`;
    btn.textContent = b.label;
    btn.addEventListener('click', () => closeModal(b.value));
    els.modalActions.appendChild(btn);
  });

  els.modalOverlay.classList.add('show');
  els.modalOverlay.setAttribute('aria-hidden', 'false');

  const firstBtn = els.modalActions.querySelector('button');
  if (firstBtn) firstBtn.focus();

  document.addEventListener('keydown', onModalKeydown);
}

function onModalKeydown(e) {
  if (e.key === 'Escape') closeModal(false);
}

function closeModal(result) {
  if (!els.modalOverlay) return;
  els.modalOverlay.classList.remove('show');
  els.modalOverlay.setAttribute('aria-hidden', 'true');
  els.modalActions.innerHTML = '';

  const resolve = modalResolve;
  modalResolve = null;
  if (resolve) resolve(result);

  document.removeEventListener('keydown', onModalKeydown);
}

function modalConfirm({
  title = 'Confirm',
  message = 'Are you sure?',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  danger = false
} = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [
        { label: cancelText, value: false },
        { label: confirmText, value: true, variant: danger ? 'danger' : 'primary' }
      ]
    });
  });
}

function modalAlert({
  title = 'Notice',
  message = '',
  okText = 'OK'
} = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [{ label: okText, value: true, variant: 'primary' }]
    });
  });
}

/* =========================================================
   5) THEME
   ========================================================= */
function applyTheme(mode) {
  if (mode === 'light') {
    els.body.classList.add('light');
  } else {
    els.body.classList.remove('light');
  }
  try { localStorage.setItem(CFG.LS_THEME, mode); } catch (_) {}
}

function initTheme() {
  let mode = CFG.DEFAULT_THEME;
  try {
    const stored = localStorage.getItem(CFG.LS_THEME);
    if (stored === 'light' || stored === 'dark') mode = stored;
  } catch (_) {}
  applyTheme(mode);

  els.themeToggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const isLight = els.body.classList.contains('light');
      applyTheme(isLight ? 'dark' : 'light');
    });
  });
}

/* =========================================================
   6) SIDEBAR COLLAPSE / MOBILE MENU
   ========================================================= */
function setSidebarCollapsed(collapsed) {
  els.body.classList.toggle('sidebar-collapsed', !!collapsed);
  try { localStorage.setItem(CFG.LS_SIDEBAR_COLLAPSED, collapsed ? '1' : '0'); } catch (_) {}
}

function initSidebarCollapse() {
  try {
    if (localStorage.getItem(CFG.LS_SIDEBAR_COLLAPSED) === '1') setSidebarCollapsed(true);
  } catch (_) {}

  if (els.sidebarCollapseBtn) els.sidebarCollapseBtn.addEventListener('click', () => setSidebarCollapsed(true));
  if (els.sidebarOpenBtn) els.sidebarOpenBtn.addEventListener('click', () => setSidebarCollapsed(false));
}

function initMobileSidebar() {
  if (els.menuBtn) {
    els.menuBtn.addEventListener('click', () => {
      els.sidebar.classList.toggle('visible');
      els.overlay.classList.toggle('visible');
    });
  }
  if (els.overlay) {
    els.overlay.addEventListener('click', () => {
      els.sidebar.classList.remove('visible');
      els.overlay.classList.remove('visible');
    });
  }
}

/* =========================================================
   7) TOOLS POPUP + TOOL STATE
   ========================================================= */
function loadToolState() {
  try {
    const raw = localStorage.getItem(CFG.LS_TOOLS);
    if (!raw) return;
    const parsed = safeJsonParse(raw, {});
    if (typeof parsed.historyTurns === 'number') {
      toolState.historyTurns = clampInt(parsed.historyTurns, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
    }
    if (['search','chat','ab'].includes(parsed.mode)) toolState.mode = parsed.mode;
  } catch (_) {}
}

function saveToolState() {
  try { localStorage.setItem(CFG.LS_TOOLS, JSON.stringify(toolState)); } catch (_) {}
}

function renderToolState() {
  if (els.historySlider) els.historySlider.value = String(toolState.historyTurns);
  if (els.historyValue) els.historyValue.textContent = String(toolState.historyTurns);
  if (els.historyHelpN) els.historyHelpN.textContent = String(toolState.historyTurns);

  const id =
    toolState.mode === 'search' ? 'mode-search' :
    toolState.mode === 'ab'     ? 'mode-ab' :
                                  'mode-chat';
  const el = document.getElementById(id);
  if (el) el.checked = true;

  if (els.modeHelp) {
    els.modeHelp.textContent =
      toolState.mode === 'search' ? 'Search the corpus without AI' :
      toolState.mode === 'ab'     ? 'A/B test two responses and select the best one' :
                                    'AI chat: normal chat mode';
  }
}

function initToolsPopup() {
  if (!els.toolsBtn || !els.toolsPopup) return;

  els.toolsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    els.toolsPopup.classList.toggle('visible');
  });

  document.addEventListener('click', (e) => {
    if (!els.toolsPopup.contains(e.target) && e.target !== els.toolsBtn) {
      els.toolsPopup.classList.remove('visible');
    }
  });

  // slider change
  if (els.historySlider) {
    els.historySlider.addEventListener('input', () => {
      toolState.historyTurns = clampInt(els.historySlider.value, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
      renderToolState();
      saveToolState();
    });
  }

  // radios change (expects value 'search'|'chat'|'ab')
  if (els.modeRadios && els.modeRadios.length) {
    els.modeRadios.forEach(r => {
      r.addEventListener('change', () => {
        if (r.checked) {
          toolState.mode = r.value;
          renderToolState();
          saveToolState();
        }
      });
    });
  }
}

/* =========================================================
   8) STATUS PANEL
   ========================================================= */
function setEndpointStatus(status) {
  if (!els.endpointDot) return;

  els.endpointDot.classList.remove('red','yellow','green');

  if (status === 'off') {
    els.endpointDot.classList.add('red');
    els.endpointLabel.textContent = 'Endpoint offline';
    els.endpointChip.textContent = 'offline';
  } else if (status === 'starting') {
    els.endpointDot.classList.add('yellow');
    els.endpointLabel.textContent = 'Endpoint starting…';
    els.endpointChip.textContent = 'starting';
  } else if (status === 'ready') {
    els.endpointDot.classList.add('green');
    els.endpointLabel.textContent = 'Endpoint ready';
    els.endpointChip.textContent = 'ready';
  } else {
    els.endpointLabel.textContent = 'Checking endpoint…';
    els.endpointChip.textContent = 'unknown';
  }
}

function setQueueStatus(queriesInLine) {
  const q = Math.max(0, parseInt(queriesInLine, 10) || 0);
  if (els.queueCountEl) els.queueCountEl.textContent = String(q);
  if (els.queueEtaEl) els.queueEtaEl.textContent = formatDuration(q * 45);
}

function pushStatusMessage(text) {
  const msg = String(text || '').trim();
  if (!msg || !els.statusMessagesEl) return;

  const placeholder = els.statusMessagesEl.querySelector('.status-message.muted');
  if (placeholder) placeholder.remove();

  const el = document.createElement('div');
  el.className = 'status-message';
  const ts = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  el.textContent = `[${ts}] ${msg}`;
  els.statusMessagesEl.prepend(el);

  const items = els.statusMessagesEl.querySelectorAll('.status-message');
  if (items.length > 1) items[items.length - 1].remove(); // keep only most recent
}

/* =========================================================
   9) FEEDBACK MODAL
   ========================================================= */
function showToast() {
  if (!els.fbToast) return;
  els.fbToast.classList.add('show');
  els.fbToast.setAttribute('aria-hidden', 'false');
  setTimeout(() => {
    els.fbToast.classList.remove('show');
    els.fbToast.setAttribute('aria-hidden', 'true');
  }, 1400);
}

function openFeedbackModal(target) {
  const isRef = target?.type === 'reference';
  feedbackTarget = target;

  // reset
  if (els.fbAccuracy) els.fbAccuracy.value = 8;
  if (els.fbStyle) els.fbStyle.value = 8;
  if (els.fbRelevance) els.fbRelevance.value = 8;
  if (els.fbComments) els.fbComments.value = '';

  // hide fields for references
  if (els.fbFieldAccuracy) els.fbFieldAccuracy.style.display = isRef ? 'none' : '';
  if (els.fbFieldStyle) els.fbFieldStyle.style.display = isRef ? 'none' : '';

  const label = isRef ? 'Reference' : 'Response';
  const snip = (target?.snippet || '').trim().replace(/\s+/g, ' ');
  const short = snip.length > 120 ? snip.slice(0, 120) + '…' : snip;
  if (els.fbMeta) els.fbMeta.textContent = `${label} ID: ${target?.id || 'n/a'}${short ? ' — ' + short : ''}`;

  els.fbOverlay.classList.add('show');
  els.fbOverlay.setAttribute('aria-hidden', 'false');

  (isRef ? els.fbRelevance : els.fbAccuracy)?.focus?.();
}

function closeFeedbackModal() {
  if (!els.fbOverlay) return;
  els.fbOverlay.classList.remove('show');
  els.fbOverlay.setAttribute('aria-hidden', 'true');
  feedbackTarget = null;
}

function saveFeedbackLocally(payload) {
  try {
    const arr = safeJsonParse(localStorage.getItem(CFG.LS_FEEDBACK) || '[]', []);
    arr.unshift(payload);
    if (arr.length > CFG.MAX_FEEDBACK_ITEMS) arr.length = CFG.MAX_FEEDBACK_ITEMS;
    localStorage.setItem(CFG.LS_FEEDBACK, JSON.stringify(arr));
  } catch (_) {}
}

async function submitFeedback() {
  if (!feedbackTarget) return;

  const isRef = feedbackTarget?.type === 'reference';

  const relevance = clamp1to10(els.fbRelevance.value);
  const accuracy = isRef ? undefined : clamp1to10(els.fbAccuracy.value);
  const style = isRef ? undefined : clamp1to10(els.fbStyle.value);

  const payload = {
    target: {
      type: feedbackTarget?.type || 'unknown',
      id: feedbackTarget?.id || null,
      snippet: feedbackTarget?.snippet || ''
    },
    ratings: {
      relevance,
      ...(isRef ? {} : { accuracy, style })
    },
    comments: (els.fbComments.value || '').trim(),
    createdAt: Date.now()
  };

  if (!relevance) {
    await modalAlert({ title: 'Missing rating', message: 'Please enter 1–10 for Relevance.' });
    return;
  }
  if (!isRef && (!accuracy || !style)) {
    await modalAlert({ title: 'Missing ratings', message: 'Please enter 1–10 for Accuracy and Style.' });
    return;
  }

  // Later: POST to Flask
  // await apiSubmitFeedback(payload);

  saveFeedbackLocally(payload);
  closeFeedbackModal();
  showToast();
}

function initFeedbackModal() {
  if (els.fbClose) els.fbClose.addEventListener('click', closeFeedbackModal);
  if (els.fbCancel) els.fbCancel.addEventListener('click', closeFeedbackModal);
  if (els.fbSubmit) els.fbSubmit.addEventListener('click', submitFeedback);

  if (els.fbOverlay) {
    els.fbOverlay.addEventListener('click', (e) => {
      if (e.target === els.fbOverlay) closeFeedbackModal();
    });
    document.addEventListener('keydown', (e) => {
      if (els.fbOverlay.classList.contains('show') && e.key === 'Escape') closeFeedbackModal();
    });
  }
}

/* =========================================================
   10) CHAT UI RENDERING
   ========================================================= */
function scrollChatToBottom() {
  if (els.chatContainer) els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
}

function appendMessage(text, role) {
  const row = document.createElement('div');
  row.className = 'message-row ' + (role === 'bot' ? 'bot' : 'user');

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar ' + (role === 'bot' ? 'bot' : 'user');
  avatar.textContent = (role === 'bot') ? 'SoT' : 'You';

  const textEl = document.createElement('div');
  textEl.className = 'message-text';
  textEl.textContent = text;

  inner.appendChild(avatar);

  if (role === 'bot') {
    const msgId = `bot_${++botMsgCounter}`;

    const contentWrap = document.createElement('div');
    contentWrap.style.display = 'flex';
    contentWrap.style.alignItems = 'flex-start';
    contentWrap.style.gap = '10px';
    contentWrap.style.width = '100%';

    textEl.style.flex = '1';

    const actions = document.createElement('div');
    actions.className = 'msg-actions';

    const commentBtn = document.createElement('button');
    commentBtn.type = 'button';
    commentBtn.className = 'comment-btn has-tooltip';
    commentBtn.dataset.tooltip = 'Add feedback';
    commentBtn.setAttribute('aria-label', 'Add feedback');
    commentBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
        stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;
    commentBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openFeedbackModal({ type: 'response', id: msgId, snippet: text });
    });

    actions.appendChild(commentBtn);
    contentWrap.appendChild(textEl);
    contentWrap.appendChild(actions);
    inner.appendChild(contentWrap);
  } else {
    inner.appendChild(textEl);
  }

  row.appendChild(inner);
  els.messagesEl.appendChild(row);
  scrollChatToBottom();
}

// A/B mode: side-by-side answers, each with its own feedback icon
function appendABMessage(aText, bText, meta = {}) {
  const row = document.createElement('div');
  row.className = 'message-row bot';

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar bot';
  avatar.textContent = 'SoT';

  const wrap = document.createElement('div');
  wrap.className = 'ab-wrap';

  function makePanel(label, text, variant) {
    const panel = document.createElement('div');
    panel.className = 'ab-panel';

    const head = document.createElement('div');
    head.className = 'ab-head';

    const lbl = document.createElement('div');
    lbl.className = 'ab-label';
    lbl.textContent = label;

    const actions = document.createElement('div');
    actions.className = 'msg-actions';

    const id = `ab_${variant}_${nowId('msg')}`;

    const commentBtn = document.createElement('button');
    commentBtn.type = 'button';
    commentBtn.className = 'comment-btn has-tooltip';
    commentBtn.dataset.tooltip = 'Add feedback';
    commentBtn.setAttribute('aria-label', 'Add feedback');
    commentBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
        stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;
    commentBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openFeedbackModal({
        type: 'response',
        id,
        snippet: `[A/B ${variant}] ${text}`
      });
    });

    actions.appendChild(commentBtn);
    head.appendChild(lbl);
    head.appendChild(actions);

    const body = document.createElement('div');
    body.className = 'message-text';
    body.textContent = text;

    panel.appendChild(head);
    panel.appendChild(body);
    return panel;
  }

  wrap.appendChild(makePanel(meta.labelA || 'Response A', aText, 'A'));
  wrap.appendChild(makePanel(meta.labelB || 'Response B', bText, 'B'));

  inner.appendChild(avatar);
  inner.appendChild(wrap);
  row.appendChild(inner);
  els.messagesEl.appendChild(row);
  scrollChatToBottom();
}

/* =========================================================
   11) REFERENCES
   ========================================================= */
function setReferences(refs) {
  const list = Array.isArray(refs) ? refs.slice(0, CFG.MAX_REFS) : [];
  els.referencesContainer.innerHTML = '';

  if (!list.length) {
    els.referencesCount.textContent = '0 items';
    els.referencesEmpty.style.display = 'block';
    return;
  }

  els.referencesEmpty.style.display = 'none';
  els.referencesCount.textContent = list.length + ' item' + (list.length === 1 ? '' : 's');

  list.forEach((ref, idx) => {
    const card = document.createElement('article');
    card.className = 'ref-card';

    const refId = `ref_${idx}_${Math.random().toString(16).slice(2)}`;

    const header = document.createElement('div');
    header.className = 'ref-header';

    const titleEl = document.createElement('div');
    titleEl.className = 'ref-title';
    titleEl.textContent = ref.title || ('Reference ' + (idx + 1));

    const right = document.createElement('div');
    right.className = 'ref-right';

    const badge = document.createElement('span');
    badge.className = 'ref-badge';
    badge.textContent = ref.source || 'Corpus';

    const cbtn = document.createElement('button');
    cbtn.type = 'button';
    cbtn.className = 'comment-btn has-tooltip';
    cbtn.dataset.tooltip = 'Add feedback';
    cbtn.setAttribute('aria-label', 'Add feedback');
    cbtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
        stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;
    cbtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openFeedbackModal({
        type: 'reference',
        id: refId,
        snippet: (ref.title ? ref.title + ' — ' : '') + (ref.snippet || '')
      });
    });

    right.appendChild(badge);
    right.appendChild(cbtn);

    header.appendChild(titleEl);
    header.appendChild(right);

    const snippetEl = document.createElement('div');
    snippetEl.className = 'ref-snippet';
    snippetEl.textContent = ref.snippet || '';

    card.appendChild(header);
    card.appendChild(snippetEl);

    if (ref.url) {
      const link = document.createElement('a');
      link.className = 'ref-link';
      link.href = ref.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'Open source';
      card.appendChild(link);
    }

    els.referencesContainer.appendChild(card);
  });
}

/* =========================================================
   12) CONVERSATION HISTORY (client-side)
   ========================================================= */
function pushTurn(userText, assistantText) {
  convoTurns.push({ user: userText, assistant: assistantText });
  if (convoTurns.length > CFG.MAX_CONVO_TURNS) convoTurns.shift();
}

function getContextTurns(n) {
  const turns = clampInt(n, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
  return convoTurns.slice(-turns);
}

/* =========================================================
   13) SAVED CONVERSATIONS (localStorage)
   ========================================================= */
function loadSavedConversations() {
  const raw = localStorage.getItem(CFG.LS_SAVED_CONVOS);
  const parsed = safeJsonParse(raw || '[]', []);
  return Array.isArray(parsed) ? parsed : [];
}

function saveSavedConversations(list) {
  try { localStorage.setItem(CFG.LS_SAVED_CONVOS, JSON.stringify(list)); } catch (_) {}
}

function makeConversationTitle(turns) {
  const first = turns?.find(t => t?.user)?.user || 'Conversation';
  const t = String(first).trim().replace(/\s+/g, ' ');
  return t.length > 42 ? t.slice(0, 42) + '…' : t;
}

function makeConversationPreview(turns) {
  if (!Array.isArray(turns) || turns.length === 0) return '';
  const last = turns[turns.length - 1];
  const txt = String(last.assistant || last.user || '').trim().replace(/\s+/g, ' ');
  return txt.length > 60 ? txt.slice(0, 60) + '…' : txt;
}

function renderRecentList() {
  const saved = loadSavedConversations();
  els.recentList.innerHTML = '';

  if (!saved.length) {
    const el = document.createElement('div');
    el.className = 'recent-item muted';
    el.textContent = 'No saved conversations yet.';
    els.recentList.appendChild(el);
    return;
  }

  saved.slice()
    .sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0))
    .forEach((item) => {
      const row = document.createElement('div');
      row.className = 'recent-preview';

      const content = document.createElement('div');
      content.className = 'recent-preview-content';

      const title = document.createElement('div');
      title.className = 'recent-preview-title';
      title.textContent = item.title || 'Conversation';

      const sub = document.createElement('div');
      sub.className = 'recent-preview-sub';
      sub.textContent = item.preview || '';

      content.appendChild(title);
      content.appendChild(sub);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'recent-delete has-tooltip';
      del.dataset.tooltip = 'Delete conversation';
      del.setAttribute('aria-label', 'Delete conversation');
      del.textContent = '×';

      del.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConversationById(item.id);
      });

      row.addEventListener('click', () => loadConversationById(item.id));

      row.appendChild(content);
      row.appendChild(del);

      els.recentList.appendChild(row);
    });
}

function saveCurrentConversation() {
  if (!Array.isArray(convoTurns) || convoTurns.length === 0) {
    pushStatusMessage('Nothing to save yet.');
    return;
  }

  const saved = loadSavedConversations();
  const item = {
    id: nowId('convo'),
    savedAt: Date.now(),
    title: makeConversationTitle(convoTurns),
    preview: makeConversationPreview(convoTurns),
    turns: convoTurns.slice()
  };

  saved.unshift(item);
  if (saved.length > CFG.MAX_SAVED_CONVOS) saved.length = CFG.MAX_SAVED_CONVOS;

  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage(`Saved conversation: "${item.title}"`);
}

function loadConversationById(id) {
  const saved = loadSavedConversations();
  const item = saved.find(x => x.id === id);
  if (!item) return;

  els.messagesEl.innerHTML = '';
  convoTurns.length = 0;
  if (els.welcomeMessage) els.welcomeMessage.style.display = 'none';

  (item.turns || []).forEach((t) => {
    if (t.user) appendMessage(t.user, 'user');
    if (t.assistant) appendMessage(t.assistant, 'bot');
    convoTurns.push({ user: t.user || '', assistant: t.assistant || '' });
  });

  pushStatusMessage(`Loaded conversation: "${item.title}"`);
}

async function deleteConversationById(id) {
  const ok = await modalConfirm({
    title: 'Delete saved conversation?',
    message: 'This will remove the saved conversation from this browser. This cannot be undone.',
    confirmText: 'Delete',
    cancelText: 'Cancel',
    danger: true
  });
  if (!ok) return;

  const saved = loadSavedConversations().filter(c => c.id !== id);
  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage('Conversation deleted.');
}

/* =========================================================
   14) INPUT AREA
   ========================================================= */
function autoResizeTextarea() {
  if (!els.chatInput) return;
  els.chatInput.style.height = 'auto';
  els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, CFG.TEXTAREA_MAX_HEIGHT) + 'px';
}

async function clearCurrentChat() {
  const hasMessages = (convoTurns && convoTurns.length) || (els.messagesEl && els.messagesEl.children.length);
  if (!hasMessages) {
    await modalAlert({ title: 'Nothing to clear', message: 'There are no messages in the current chat yet.' });
    return;
  }

  const ok = await modalConfirm({
    title: 'Clear current chat?',
    message: 'This clears the current chat on this page. Saved chats are not affected.',
    confirmText: 'Clear',
    cancelText: 'Cancel',
    danger: true
  });
  if (!ok) return;

  els.messagesEl.innerHTML = '';
  convoTurns.length = 0;
  if (els.welcomeMessage) els.welcomeMessage.style.display = '';
  setReferences([]);
  pushStatusMessage('Current chat cleared.');
}

/* =========================================================
   15) FLASK API WRAPPERS (call these once app.py is ready)
   ========================================================= */
async function apiChat(payload) {
  const res = await fetch(CFG.API.CHAT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`apiChat failed: ${res.status}`);
  return await res.json();
}

async function apiSubmitFeedback(payload) {
  const res = await fetch(CFG.API.FEEDBACK, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`apiSubmitFeedback failed: ${res.status}`);
  return await res.json();
}

async function apiStatus() {
  const res = await fetch(CFG.API.STATUS, { method: 'GET' });
  if (!res.ok) throw new Error(`apiStatus failed: ${res.status}`);
  return await res.json();
}

async function apiQueue() {
  const res = await fetch(CFG.API.QUEUE, { method: 'GET' });
  if (!res.ok) throw new Error(`apiQueue failed: ${res.status}`);
  return await res.json();
}

/* =========================================================
   16) CHAT HANDLING (submit + render)
   ========================================================= */
async function handleChatSubmit(e) {
  e.preventDefault();
  const text = (els.chatInput.value || '').trim();
  if (!text) return;

  if (els.welcomeMessage) els.welcomeMessage.style.display = 'none';

  appendMessage(text, 'user');
  els.chatInput.value = '';
  autoResizeTextarea();

  const contextTurns = getContextTurns(toolState.historyTurns);

  const payload = {
    message: text,
    mode: toolState.mode,
    history_turns: toolState.historyTurns,
    context: contextTurns
  };

  // --- STUB behavior (replace with apiChat(payload) when ready) ---
  setTimeout(() => {
    if (toolState.mode === 'search') {
      const bot = 'Search-only mode (stub): will return references/snippets without a generated answer.';
      appendMessage(bot, 'bot');
      setReferences([]);
      pushTurn(text, bot);
      return;
    }

    if (toolState.mode === 'ab') {
      const a = 'A/B (stub) — Answer A will appear here.';
      const b = 'A/B (stub) — Answer B will appear here.';
      appendABMessage(a, b);
      setReferences([]);
      // store a single assistant string for saved convos
      pushTurn(text, `Response A:\n${a}\n\nResponse B:\n${b}`);
      return;
    }

    const reply = 'AI chat mode (stub): connect to Flask/HF and include selected history turns in the request.';
    appendMessage(reply, 'bot');
    setReferences([]);
    pushTurn(text, reply);
  }, CFG.STUB_DELAY_MS);

  // --- Real behavior (uncomment when Flask is ready) ---
  // try {
  //   const data = await apiChat(payload);
  //   // Expected examples:
  //   // chat:   { reply: "...", references: [...] }
  //   // search: { references: [...] }
  //   // ab:     { a: "...", b: "...", references: [...] }
  //   if (toolState.mode === 'ab') {
  //     appendABMessage(data.a || '', data.b || '', { labelA: data.labelA, labelB: data.labelB });
  //     pushTurn(text, `Response A:\n${data.a || ''}\n\nResponse B:\n${data.b || ''}`);
  //   } else if (toolState.mode === 'search') {
  //     appendMessage(data.message || '(search results)', 'bot');
  //     pushTurn(text, data.message || '(search results)');
  //   } else {
  //     appendMessage(data.reply || '', 'bot');
  //     pushTurn(text, data.reply || '');
  //   }
  //   setReferences(Array.isArray(data.references) ? data.references : []);
  // } catch (err) {
  //   appendMessage('Error contacting server. Please try again.', 'bot');
  //   pushStatusMessage(String(err?.message || err));
  // }
}

/* =========================================================
   17) ABOUT MODAL
   ========================================================= */
function initAboutModal() {
  if (!els.aboutBtn) return;
  els.aboutBtn.addEventListener('click', async () => {
    await modalAlert({
      title: 'About Seeds of Truth',
      message:
        'Seeds of Truth is an AI research chat for open testing. Responses may be inaccurate. ' +
        'Please verify important information. Your feedback helps improve quality.',
      okText: 'Close'
    });
  });
}

/* =========================================================
   18) INIT / WIRING (bottom)
   ========================================================= */
function initDom() {
  els.body = document.body;

  // core chat
  els.chatForm = document.getElementById('chat-form');
  els.chatInput = document.getElementById('chat-input');
  els.messagesEl = document.getElementById('messages');
  els.chatContainer = document.getElementById('chat-container');
  els.welcomeMessage = document.getElementById('welcome-message');

  // theme + sidebar
  els.themeToggleButtons = Array.from(document.querySelectorAll('[data-theme-toggle]'));
  els.sidebar = document.getElementById('sidebar');
  els.menuBtn = document.getElementById('menu-btn');
  els.overlay = document.getElementById('overlay');
  els.sidebarOpenBtn = document.getElementById('sidebar-open-btn');
  els.sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');

  // tools
  els.toolsBtn = document.getElementById('tools-btn');
  els.toolsPopup = document.getElementById('tools-popup');

  // refs
  els.referencesContainer = document.getElementById('references-container');
  els.referencesCount = document.getElementById('references-count');
  els.referencesEmpty = document.getElementById('references-empty');

  // tool controls
  els.historySlider = document.getElementById('history-slider');
  els.historyValue = document.getElementById('history-value');
  els.historyHelpN = document.getElementById('history-help-n');
  els.modeHelp = document.getElementById('mode-help');
  els.modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));

  // status panel
  els.endpointDot = document.getElementById('endpoint-dot');
  els.endpointLabel = document.getElementById('endpoint-label');
  els.endpointChip = document.getElementById('endpoint-chip');
  els.queueCountEl = document.getElementById('queue-count');
  els.queueEtaEl = document.getElementById('queue-eta');
  els.statusMessagesEl = document.getElementById('status-messages');

  // saved convos
  els.saveConvoBtn = document.getElementById('save-convo-btn');
  els.recentList = document.getElementById('recent-list');

  // trash chat
  els.trashChatBtn = document.getElementById('trash-chat-btn');

  // about
  els.aboutBtn = document.getElementById('about-btn');

  // feedback modal
  els.fbOverlay = document.getElementById('fb-overlay');
  els.fbClose = document.getElementById('fb-close');
  els.fbCancel = document.getElementById('fb-cancel');
  els.fbSubmit = document.getElementById('fb-submit');
  els.fbMeta = document.getElementById('fb-meta');
  els.fbAccuracy = document.getElementById('fb-accuracy');
  els.fbStyle = document.getElementById('fb-style');
  els.fbRelevance = document.getElementById('fb-relevance');
  els.fbComments = document.getElementById('fb-comments');
  els.fbToast = document.getElementById('fb-toast');
  els.fbFieldAccuracy = document.getElementById('fb-field-accuracy');
  els.fbFieldStyle = document.getElementById('fb-field-style');

  // custom modal
  els.modalOverlay = document.getElementById('modal-overlay');
  els.modalTitle = document.getElementById('modal-title');
  els.modalMessage = document.getElementById('modal-message');
  els.modalActions = document.getElementById('modal-actions');
  els.modalCloseBtn = document.getElementById('modal-close');
}

function initWiring() {
  // modal wiring
  if (els.modalOverlay) {
    els.modalOverlay.addEventListener('click', (e) => {
      if (e.target === els.modalOverlay) closeModal(false);
    });
  }
  if (els.modalCloseBtn) {
    els.modalCloseBtn.addEventListener('click', () => closeModal(false));
  }

  // chat
  if (els.chatForm) els.chatForm.addEventListener('submit', handleChatSubmit);
  if (els.chatInput) {
    els.chatInput.addEventListener('input', autoResizeTextarea);
    els.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        els.chatForm.requestSubmit();
      }
    });
  }

  // save convo
  if (els.saveConvoBtn) els.saveConvoBtn.addEventListener('click', saveCurrentConversation);

  // trash chat
  if (els.trashChatBtn) els.trashChatBtn.addEventListener('click', clearCurrentChat);
}

function init() {
  initDom();

  // sanity: required nodes
  if (!els.chatForm || !els.chatInput || !els.messagesEl) {
    console.warn('Seeds of Truth app.js: required chat elements not found.');
    return;
  }

  // tool state
  loadToolState();
  renderToolState();

  // theme + sidebar + tools
  initTheme();
  initSidebarCollapse();
  initMobileSidebar();
  initToolsPopup();

  // feedback + about + wiring
  initFeedbackModal();
  initAboutModal();
  initWiring();

  // initial UI
  renderRecentList();
  autoResizeTextarea();

  // expose for backend wiring / debugging
  window.setReferences = setReferences;
  window.openFeedbackModal = openFeedbackModal;
  window.pushStatusMessage = pushStatusMessage;

  // Optional: initialize status display
  setEndpointStatus('unknown');
  setQueueStatus(0);
}

document.addEventListener('DOMContentLoaded', init);
