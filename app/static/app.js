const state = {
  currentTab: "agent",
  lastBatchId: null,
  agentTimeline: [],
  agentPendingApproval: null,
  agentThreadId: "",
  agentBusy: false,
  agentViewVersion: 0,
  agentStateAbortController: null,
  agentStreamAbortController: null,
  currentAssistantIndex: null,
  selectedFiles: [],
  experimentMode: "standard",
  retrievalMode: "standard",
  contextLastFocused: null,
  agentSources: [],
  documentView: "documents",
  webSearchEnabled: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function onClick(selector, handler) {
  const element = $(selector);
  if (element) element.addEventListener("click", handler);
}

const RAGAS_METRIC_LABELS = {
  faithfulness: "忠实度 (Faithfulness)",
  answer_relevancy: "回答相关性 (Answer Relevancy)",
  context_precision: "上下文精度 (Context Precision)",
  llm_context_precision_with_reference: "上下文精度 (Context Precision)",
  context_entity_recall: "上下文实体召回 (Context Entity Recall)",
  "noise_sensitivity(mode=relevant)": "噪声敏感度 (Noise Sensitivity)",
  noise_sensitivity: "噪声敏感度 (Noise Sensitivity)",
  context_recall: "上下文召回 (Context Recall)",
};

const TOOL_LABELS = {
  search_knowledge_base: "检索知识库",
  search_web: "联网搜索",
  list_documents: "查看文档列表",
  get_document_detail: "查看文档详情",
  get_system_status: "查看系统状态",
  list_upload_batches: "查看上传批次",
  get_upload_batch_detail: "查看批次详情",
  retry_upload_item: "重试上传文件",
  reindex_document: "重建文档索引",
  delete_document: "删除文档",
};

let noticeTimer = null;
let lastRagasMode = "baseline";
let lastRagasStrategy = "dense";

function formatRagasMetricName(name) {
  return RAGAS_METRIC_LABELS[name] || `${name} (${name})`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getErrorMessage(error, fallback = "请求失败") {
  const value = error instanceof Error ? error.message : error;
  if (value && typeof value === "object") {
    if (value.detail) return getErrorMessage(value.detail, fallback);
    if (value.message) return getErrorMessage(value.message, fallback);
    if (value.msg) return getErrorMessage(value.msg, fallback);
    if (value.error) return getErrorMessage(value.error, fallback);
    try {
      return JSON.stringify(value);
    } catch {
      return fallback;
    }
  }
  let message = String(value ?? "").trim();
  // 代理或静态服务器有时会返回 HTML 错误页，界面只需要可读的摘要。
  if (/<\/?[a-z][\s\S]*>/i.test(message)) {
    try {
      message = new DOMParser().parseFromString(message, "text/html").body.textContent || message;
    } catch {
      message = message.replace(/<[^>]*>/g, " ");
    }
  }
  message = message.replace(/\s+/g, " ").trim();
  return message ? (message.length > 420 ? `${message.slice(0, 420)}…` : message) : fallback;
}

function showNotice(message, type = "success") {
  const notice = $("#notice");
  if (!notice) return;
  window.clearTimeout(noticeTimer);
  notice.hidden = false;
  notice.className = `notice ${type}`;
  notice.textContent = getErrorMessage(message);
  noticeTimer = window.setTimeout(() => {
    notice.hidden = true;
  }, type === "error" ? 8000 : 4800);
}

function hideNotice() {
  const notice = $("#notice");
  if (notice) notice.hidden = true;
}

function setLoading(button, loading, label) {
  if (!button) return;
  if (loading) {
    button.dataset.originalHtml = button.innerHTML;
    button.textContent = label || "处理中...";
    button.disabled = true;
    return;
  }
  button.innerHTML = button.dataset.originalHtml || button.innerHTML;
  button.disabled = false;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail || payload : payload;
    throw new Error(getErrorMessage(detail, `HTTP ${response.status}`));
  }
  return payload;
}

async function consumeSse(response, onEvent) {
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    const detail = typeof payload === "object" ? payload.detail || payload : payload;
    throw new Error(getErrorMessage(detail, `HTTP ${response.status}`));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const processBuffer = (flush = false) => {
    const events = buffer.split("\n\n");
    buffer = flush ? "" : events.pop() || "";
    for (const event of events) {
      const line = event.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const data = line.slice(6);
      if (data === "[DONE]") continue;
      try {
        onEvent(JSON.parse(data));
      } catch {
        onEvent({ event: "error", message: data });
      }
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    processBuffer();
  }
  buffer += decoder.decode();
  if (buffer.trim()) processBuffer(true);
}

function parseIntInput(selector, fallback) {
  const element = $(selector);
  if (!element || element.value === "") return fallback;
  const parsed = Number.parseInt(element.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOptionalInt(selector) {
  const element = $(selector);
  if (!element || !element.value.trim()) return null;
  const parsed = Number.parseInt(element.value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDuration(durationMs) {
  const value = Number(durationMs || 0);
  return value ? `${value} ms` : "-";
}

const UPLOAD_MAX_FILES = 10;
const UPLOAD_MAX_BYTES = 50 * 1024 * 1024;
const SUPPORTED_UPLOAD_EXTENSIONS = new Set([
  "bmp", "doc", "docx", "gif", "html", "jp2", "jpeg", "jpg", "md", "pdf",
  "png", "ppt", "pptx", "txt", "webp",
]);

function getFileExtension(filename) {
  const parts = String(filename || "").toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function validateUploadFile(file) {
  if (!file) return "文件对象无效";
  const extension = getFileExtension(file.name);
  if (!SUPPORTED_UPLOAD_EXTENSIONS.has(extension)) return `不支持 .${extension || "未知"} 格式`;
  if (file.size > UPLOAD_MAX_BYTES) return "文件超过 50 MB 限制";
  return "";
}

function formatUploadStatus(status) {
  const labels = {
    running: "处理中",
    completed: "已完成",
    partial_success: "部分成功",
    failed: "失败",
    queued: "排队中",
    saving: "保存中",
    parsing: "解析中",
    splitting: "切分中",
    embedding: "向量化中",
    indexed: "已入库",
    duplicate: "重复跳过",
  };
  const label = labels[status] || status || "未知";
  return `<span class="status-badge status-${escapeHtml(status || "unknown")}">${escapeHtml(label)}</span>`;
}

function setStatusHealth(healthy) {
  const dot = $("#statusDot");
  const sidebarDot = $("#sidebarHealth .health-dot");
  [dot, sidebarDot].forEach((element) => {
    if (!element) return;
    element.style.background = healthy ? "var(--success)" : "var(--danger)";
  });
}

async function refreshStatus() {
  try {
    const health = await apiFetch("/api/health");
    const status = await apiFetch("/api/status");
    const ready = Boolean(health.agent?.ready);
    const healthLabel = ready ? "Agent 就绪" : health.status || "服务正常";
    $("#healthText").textContent = healthLabel;
    $("#sidebarHealthText").textContent = healthLabel;
    $("#documentCount").textContent = status.document_count;
    $("#chunkCount").textContent = status.chunk_count;
    $("#embeddingProvider").textContent = status.embedding_provider;
    setStatusHealth(health.status === "ok");
  } catch (error) {
    $("#healthText").textContent = "服务异常";
    $("#sidebarHealthText").textContent = "服务异常";
    setStatusHealth(false);
    showNotice(`状态加载失败：${getErrorMessage(error)}`, "error");
  }
}

function markdownToHtml(value, sources = [], messageIndex = null) {
  const text = String(value ?? "").replaceAll("\r\n", "\n");
  if (!text) return "";
  if (!window.marked || !window.DOMPurify) {
    return `<p>${escapeHtml(text).replaceAll("\n", "<br />")}</p>`;
  }

  const sourceTokens = [];
  const tokenizedText = text.replace(/\[(\d+)\]/g, (match, number) => {
    const index = Number(number);
    if (index < 1 || index > sources.length) return match;
    const token = `RAG_SOURCE_REFERENCE_${sourceTokens.length}_END`;
    sourceTokens.push({ token, index });
    return token;
  });
  const rawHtml = window.marked.parse(tokenizedText, {
    gfm: true,
    breaks: true,
    headerIds: false,
    mangle: false,
  });
  let cleanHtml = window.DOMPurify.sanitize(rawHtml, {
    USE_PROFILES: { html: true },
  });
  for (const { token, index } of sourceTokens) {
    cleanHtml = cleanHtml.replace(
      token,
      `<button class="source-ref" type="button" data-source-index="${index}" data-message-index="${messageIndex ?? ""}" aria-label="查看来源 ${index}">[${index}]</button>`
    );
  }
  return cleanHtml;
}

function makeId(prefix) {
  if (window.crypto?.randomUUID) return `${prefix}-${window.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadSessionHistory() {
  try {
    const value = JSON.parse(localStorage.getItem("knowledge_agent_sessions") || "[]");
    if (!Array.isArray(value)) return [];
    return value.map(normalizeSessionRecord).filter(Boolean);
  } catch {
    return [];
  }
}

function saveSessionHistory(sessions) {
  try {
    const normalized = (Array.isArray(sessions) ? sessions : [])
      .map(normalizeSessionRecord)
      .filter(Boolean);
    localStorage.setItem("knowledge_agent_sessions", JSON.stringify(normalized.slice(0, 20)));
  } catch {
    // 浏览器存储不可用时，当前会话仍可继续使用。
  }
}

function normalizeSessionRecord(session) {
  if (!session || typeof session !== "object") return null;
  const threadId = normalizeText(session.threadId);
  if (!threadId) return null;
  const title = normalizeText(session.title) || "新会话";
  const parsedTime = Number(session.updatedAt);
  const updatedAt = Number.isFinite(parsedTime) && parsedTime > 0 ? parsedTime : Date.now();
  const allowedStatuses = new Set(["idle", "running", "awaiting_approval", "completed", "failed"]);
  const status = allowedStatuses.has(normalizeText(session.status)) ? normalizeText(session.status) : "idle";
  return { ...session, threadId, title, updatedAt, status };
}

function upsertSessionMeta(threadId, patch = {}) {
  const normalizedThreadId = normalizeText(threadId);
  if (!normalizedThreadId) return;
  const sessions = loadSessionHistory();
  const existing = sessions.find((session) => session.threadId === normalizedThreadId);
  const next = normalizeSessionRecord({
    threadId: normalizedThreadId,
    title: existing?.title || "新会话",
    updatedAt: existing?.updatedAt || Date.now(),
    status: existing?.status || "idle",
    ...patch,
  });
  if (!next) return;
  const rest = sessions.filter((session) => session.threadId !== normalizedThreadId);
  saveSessionHistory([next, ...rest]);
  renderSessionHistory();
}

function sessionTitleFromInput(input) {
  const normalized = String(input || "").replace(/\s+/g, " ").trim();
  return normalized.length > 24 ? `${normalized.slice(0, 24)}…` : normalized || "新会话";
}

function renderSessionHistory() {
  const list = $("#sessionHistoryList");
  if (!list) return;
  const sessions = loadSessionHistory();
  if (!sessions.length) {
    list.innerHTML = `<span class="session-empty">暂无历史会话</span>`;
    return;
  }
  list.innerHTML = sessions.slice(0, 6).map((session) => `
    <div class="session-item-row">
      <button class="session-item ${session.threadId === state.agentThreadId ? "active" : ""}" type="button" data-thread-id="${escapeHtml(session.threadId)}" title="${escapeHtml(session.title || "新会话")}">
        <span class="session-status-dot status-${escapeHtml(session.status || "idle")}" aria-hidden="true"></span>
        <span class="session-item-copy"><strong>${escapeHtml(session.title || "新会话")}</strong><small>${escapeHtml(formatRelativeTime(session.updatedAt))}</small></span>
      </button>
      <button class="session-delete" type="button" data-session-delete="${escapeHtml(session.threadId)}" title="删除会话" aria-label="删除会话 ${escapeHtml(session.title || "新会话")}">
        <svg class="icon" aria-hidden="true"><use href="/static/icons.svg#trash-2"></use></svg>
      </button>
    </div>
  `).join("");
}

function removeSessionMeta(threadId) {
  const normalizedThreadId = normalizeText(threadId);
  if (!normalizedThreadId) return;
  const sessions = loadSessionHistory().filter((session) => session.threadId !== normalizedThreadId);
  saveSessionHistory(sessions);
  try {
    localStorage.removeItem(`knowledge_agent_messages_${normalizedThreadId}`);
  } catch {
    // 浏览器存储不可用时，服务端 checkpoint 仍然已经删除。
  }
  renderSessionHistory();
}

function formatRelativeTime(timestamp) {
  const value = Number(timestamp || 0);
  if (!value) return "刚刚";
  const seconds = Math.max(0, Math.floor((Date.now() - value) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return `${Math.floor(seconds / 86400)} 天前`;
}

function initAgentThread() {
  const storageKey = "knowledge_agent_current_thread";
  let threadId = normalizeText(localStorage.getItem(storageKey));
  if (!threadId) {
    threadId = makeId("agent");
    localStorage.setItem(storageKey, threadId);
  }
  state.agentThreadId = threadId;
  state.webSearchEnabled = localStorage.getItem("knowledge_agent_web_search") === "true";
  $("#agentThreadId").value = threadId;
  upsertSessionMeta(threadId);
  updateThreadLabel();
  renderSessionHistory();
}

function invalidateAgentView({ abortStream = false } = {}) {
  state.agentViewVersion += 1;
  state.agentStateAbortController?.abort();
  state.agentStateAbortController = null;
  if (abortStream) {
    state.agentStreamAbortController?.abort();
    state.agentStreamAbortController = null;
  }
  return state.agentViewVersion;
}

function isCurrentAgentContext(context) {
  return Boolean(
    context &&
    context.threadId === state.agentThreadId &&
    context.version === state.agentViewVersion
  );
}

function selectAgentThread(threadId) {
  if (!threadId || threadId === state.agentThreadId) return;
  activateAgentThread(threadId);
}

function updateThreadLabel() {
  const id = state.agentThreadId || "agent";
  const shortId = id.length > 18 ? `${id.slice(0, 8)}...${id.slice(-5)}` : id;
  $("#threadLabel").textContent = `当前会话 · ${shortId}`;
}

function conversationStorageKey() {
  const threadId = normalizeText(state.agentThreadId) || "uninitialized";
  return `knowledge_agent_messages_${threadId}`;
}

function normalizeText(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : String(value);
  const normalized = text.trim();
  return !normalized || normalized.toLowerCase() === "null" || normalized.toLowerCase() === "undefined"
    ? ""
    : normalized;
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.filter((source) => {
    if (!source || typeof source !== "object") return false;
    const isWeb = source.source_type === "web" || Boolean(source.url);
    return isWeb
      ? Boolean(normalizeText(source.title) || normalizeText(source.url))
      : Boolean(normalizeText(source.filename) || normalizeText(source.document_id));
  });
}

function loadConversation() {
  try {
    const value = JSON.parse(localStorage.getItem(conversationStorageKey()) || "[]");
    if (!Array.isArray(value)) return [];
    return value.filter((message) => message && ["user", "assistant"].includes(message.role) && normalizeText(message.content));
  } catch {
    return [];
  }
}

function saveConversation(messages) {
  try {
    localStorage.setItem(conversationStorageKey(), JSON.stringify(messages.slice(-40)));
  } catch {
    // 浏览器存储不可用时，当前页面仍然可以继续对话。
  }
}

function renderConversation() {
  const stream = $("#agentConversation");
  if (!stream) return;
  let messageList = $("#agentMessageList");
  // 兼容旧页面缓存：如果旧 HTML 没有消息容器，动态补上，避免清空消息时误删欢迎页。
  if (!messageList) {
    messageList = document.createElement("div");
    messageList.className = "conversation-message-list";
    messageList.id = "agentMessageList";
    stream.append(messageList);
  }
  const messages = loadConversation();
  const welcome = $("#agentWelcome");
  if (!messages.length) {
    messageList.innerHTML = "";
    if (welcome) welcome.hidden = false;
    return;
  }

  const renderedMessages = messages.map((message, messageIndex) => {
    const isUser = message.role === "user";
    const messageContent = normalizeText(message.content);
    const messageSources = normalizeSources(message.sources);
    const content = isUser
      ? escapeHtml(messageContent).replaceAll("\n", "<br />")
      : markdownToHtml(messageContent, messageSources, messageIndex);
    if (!content && !message.pending) return "";
    return `
      <article class="message ${isUser ? "user" : "assistant"} ${message.pending ? "pending" : ""}">
        <span class="message-label">${isUser ? "你" : "Agent"}</span>
        <div class="message-bubble">${content || (message.pending ? "正在处理..." : "")}</div>
        ${message.meta ? `<span class="message-meta">${escapeHtml(message.meta)}</span>` : ""}
      </article>
    `;
  }).filter(Boolean).join("");

  if (!renderedMessages) {
    messageList.innerHTML = "";
    if (welcome) welcome.hidden = false;
    return;
  }
  messageList.innerHTML = renderedMessages;
  if (welcome) welcome.hidden = true;

  stream.scrollTop = stream.scrollHeight;
}

function appendConversationMessage(role, content, options = {}) {
  const normalizedContent = normalizeText(content);
  if (!normalizedContent) return null;
  const messages = loadConversation();
  messages.push({ role, content: normalizedContent, ...options });
  saveConversation(messages);
  state.currentAssistantIndex = role === "assistant" ? messages.length - 1 : null;
  renderConversation();
  return messages.length - 1;
}

function updateCurrentAssistant(content, options = {}) {
  const messages = loadConversation();
  let index = state.currentAssistantIndex;
  const normalizedContent = normalizeText(content);
  if (index == null || !messages[index] || messages[index].role !== "assistant") {
    index = appendConversationMessage("assistant", normalizedContent, options);
    state.currentAssistantIndex = index;
    return;
  }
  if (normalizedContent) messages[index] = { ...messages[index], content: normalizedContent, ...options };
  else messages[index] = { ...messages[index], ...options };
  saveConversation(messages);
  renderConversation();
}

function resetAgentOutput(clearConversation = true) {
  state.agentTimeline = [];
  state.agentPendingApproval = null;
  state.agentSources = [];
  state.currentAssistantIndex = null;
  if (clearConversation) {
    localStorage.removeItem(conversationStorageKey());
    upsertSessionMeta(state.agentThreadId, { title: "新会话", status: "idle", updatedAt: Date.now() });
  }
  $("#agentAnswer").textContent = "";
  renderConversation();
  renderAgentTimeline();
  renderAgentApproval(null);
  setAgentBusy(false);
  setAgentStatus("准备就绪，可以开始提问。");
}

function activateAgentThread(threadId, { loadState = true } = {}) {
  const version = invalidateAgentView({ abortStream: true });
  state.agentThreadId = threadId;
  state.agentBusy = false;
  localStorage.setItem("knowledge_agent_current_thread", threadId);
  $("#agentThreadId").value = threadId;
  updateThreadLabel();
  state.agentTimeline = [];
  state.agentPendingApproval = null;
  state.agentSources = [];
  state.currentAssistantIndex = null;
  $("#agentAnswer").textContent = "";
  renderSessionHistory();
  renderAgentTimeline();
  renderAgentApproval(null);
  renderConversation();
  setAgentBusy(false);
  if (loadState) {
    setAgentStatus("正在恢复会话状态...");
    loadAgentState(threadId, version);
  } else {
    setAgentStatus("准备就绪，可以开始提问。");
  }
  return version;
}

function setAgentStatus(message) {
  const box = $("#agentStatus");
  if (box) box.textContent = message;
}

function setAgentBusy(busy) {
  state.agentBusy = busy;
  const input = $("#agentInput");
  const button = $("#agentRunBtn");
  if (input) input.disabled = busy || Boolean(state.agentPendingApproval);
  if (button) {
    button.disabled = busy || Boolean(state.agentPendingApproval);
    button.innerHTML = busy
      ? "<span class=\"button-symbol\" aria-hidden=\"true\">…</span> 处理中"
      : "<span class=\"button-symbol\" aria-hidden=\"true\"><svg class=\"icon\"><use href=\"/static/icons.svg#send\"></use></svg></span> 发送";
  }
}

function selectContextView(viewId) {
  $$(".context-tab").forEach((button) => {
    const active = button.dataset.context === viewId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  $$(".context-view").forEach((view) => {
    view.hidden = view.id !== viewId;
    view.classList.toggle("active", view.id === viewId);
  });
}

function openContextPanel(viewId = "sourcesContext") {
  selectContextView(viewId);
  const panel = $("#contextPanel");
  const workspace = $("#agentWorkspace");
  const scrim = $("#contextScrim");
  state.contextLastFocused = document.activeElement;
  panel?.classList.remove("is-collapsed");
  panel?.classList.add("open");
  workspace?.classList.add("context-expanded");
  if (scrim) scrim.hidden = false;
  document.body.classList.add("context-open");
  window.setTimeout(() => {
    const target = viewId === "approvalContext" && state.agentPendingApproval
      ? $("#approveAgentBtn")
      : $("#closeContextBtn");
    target?.focus();
  }, 0);
}

function closeContextPanel({ restoreFocus = true } = {}) {
  const panel = $("#contextPanel");
  const workspace = $("#agentWorkspace");
  const scrim = $("#contextScrim");
  panel?.classList.remove("open");
  panel?.classList.add("is-collapsed");
  workspace?.classList.remove("context-expanded");
  if (scrim) scrim.hidden = true;
  document.body.classList.remove("context-open");
  if (restoreFocus && state.contextLastFocused instanceof HTMLElement) state.contextLastFocused.focus();
}

function isContextOpen() {
  return Boolean($("#contextPanel")?.classList.contains("open"));
}

function trapContextFocus(event) {
  if (event.key !== "Tab" || !isContextOpen()) return;
  const panel = $("#contextPanel");
  const focusable = Array.from(panel.querySelectorAll("button, [href], input, textarea, select, summary, [tabindex]:not([tabindex=\"-1\"])"))
    .filter((element) => !element.disabled && !element.hidden && element.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function renderSources(boxSelector, sources = [], options = {}) {
  const box = $(boxSelector);
  if (!box) return;
  sources = normalizeSources(sources);
  const empty = options.emptySelector ? $(options.emptySelector) : null;
  if (empty) empty.hidden = sources.length > 0;
  if (options.countSelector) $(options.countSelector).textContent = sources.length;
  if (!sources.length) {
    box.innerHTML = "";
    return;
  }

  box.innerHTML = sources.map((source, index) => {
    const isWeb = source.source_type === "web" || source.url;
    const title = isWeb
      ? (normalizeText(source.title) || normalizeText(source.url) || "网页来源")
      : (normalizeText(source.filename) || normalizeText(source.document_id) || "知识库文档");
    const sourceUrl = normalizeText(source.url);
    const strategy = normalizeText(source.retrieval_strategy) || "dense";
    const location = isWeb
      ? (sourceUrl ? `<a href="${safeHttpUrl(sourceUrl)}" target="_blank" rel="noreferrer">打开网页</a>` : "联网来源")
      : `切片 ${escapeHtml(source.chunk_index ?? "-")} · ${escapeHtml(strategy)}`;
    return `
    <details class="source-card" id="source-card-${index + 1}">
      <summary class="source-summary">
        <strong>${index + 1}. ${escapeHtml(title)}</strong>
        <span class="source-meta">${isWeb ? "联网来源 · " : "知识库来源 · "}${location}</span>
      </summary>
      <pre>${escapeHtml(normalizeText(source.content_preview) || normalizeText(source.content))}</pre>
    </details>
  `; }).join("");
}

function safeHttpUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return ["http:", "https:"].includes(url.protocol) ? escapeHtml(url.href) : "#";
  } catch {
    return "#";
  }
}

function renderAgentSources(sources = []) {
  state.agentSources = normalizeSources(sources);
  renderSources("#agentSources", state.agentSources, {
    emptySelector: "#sourcesEmpty",
    countSelector: "#sourceCount",
  });
  $("#railSourceCount").textContent = state.agentSources.length;
  $("[data-context=\"sourcesContext\"]")?.classList.toggle("has-items", state.agentSources.length > 0);
  renderConversation();
}

function getToolLabel(toolName) {
  return TOOL_LABELS[toolName] || toolName || "未知工具";
}

function renderAgentTimeline() {
  const box = $("#agentTimeline");
  const empty = $("#timelineEmpty");
  if (!box) return;
  const calls = state.agentTimeline.filter((item) => item.type === "tool" || item.type === "call");
  if (empty) empty.hidden = calls.length > 0;
  $("#traceCount").textContent = calls.length;
  $("#railTraceCount").textContent = calls.length;
  $("[data-context=\"timelineContext\"]")?.classList.toggle("has-items", calls.length > 0);
  if (!state.agentTimeline.length) {
    box.innerHTML = "";
    return;
  }

  box.innerHTML = state.agentTimeline.map((item) => {
    const pending = item.ok === undefined;
    const statusLabel = pending
      ? (item.requires_approval ? "等待审批" : "执行中")
      : (item.ok ? "已完成" : "执行失败");
    const details = item.result === undefined
      ? "查看参数"
      : "查看参数与结果";
    return `
      <div class="timeline-item ${item.ok === false ? "tool-error" : ""} ${pending ? "tool-pending" : ""}">
        <div class="timeline-item-heading">
          <strong>${escapeHtml(getToolLabel(item.tool_name))}</strong>
          <span class="timeline-status">${escapeHtml(statusLabel)}</span>
        </div>
        <span>${escapeHtml(item.summary || (pending ? "正在执行工具" : "工具已返回结果"))}</span>
        <details class="timeline-details"><summary>${details}</summary><pre>${escapeHtml(JSON.stringify({ args: item.args || {}, result: item.result }, null, 2))}</pre></details>
      </div>
    `;
  }).join("");
}

function renderAgentApproval(pending) {
  const card = $("#agentApprovalCard");
  const empty = $("#approvalEmpty");
  const tab = $(".approval-tab");
  state.agentPendingApproval = pending || null;
  if (card) card.hidden = !pending;
  if (empty) empty.hidden = Boolean(pending);
  if (tab) tab.classList.toggle("has-pending", Boolean(pending));
  $("#approvalCount").textContent = pending ? "1" : "0";
  $("#railApprovalCount").textContent = pending ? "1" : "0";
  $("[data-context=\"approvalContext\"]")?.classList.toggle("has-items", Boolean(pending));
  if (!pending) {
    setAgentBusy(state.agentBusy);
    return;
  }
  $("#approvalToolName").textContent = getToolLabel(pending.tool_name);
  $("#approvalSummary").textContent = pending.summary || "-";
  $("#approvalArgs").textContent = JSON.stringify(pending.args || {}, null, 2);
  $("#approvalRisk").textContent = pending.risk || "需要人工确认";
  $("#approvalId").textContent = pending.approval_id || "-";
  setAgentBusy(false);
  $("#agentInput").disabled = true;
  $("#agentRunBtn").disabled = true;
  window.setTimeout(() => $("#approveAgentBtn")?.focus(), 0);
}

function syncAgentAnswer(agentState) {
  const answer = normalizeText(agentState.last_answer);
  const sources = normalizeSources(agentState.sources);
  const messages = loadConversation();
  const existingAnswerIndex = [...messages].map((message, index) => ({ message, index }))
    .reverse()
    .find(({ message }) => message.role === "assistant" && normalizeText(message.content) === answer)?.index;
  const lastAssistantIndex = [...messages].map((message, index) => ({ message, index }))
    .reverse()
    .find(({ message }) => message.role === "assistant")?.index;

  if (!answer) {
    renderConversation();
    return;
  }

  const targetIndex = lastAssistantIndex !== undefined && messages[lastAssistantIndex].pending
    ? lastAssistantIndex
    : existingAnswerIndex;
  if (targetIndex !== undefined) {
    const target = messages[targetIndex];
    messages[targetIndex] = {
        ...target,
        content: answer,
        pending: false,
        sources,
        runId: agentState.run_id || target.runId || "",
      };
      saveConversation(messages);
      state.currentAssistantIndex = targetIndex;
      renderConversation();
      return;
  }

  state.currentAssistantIndex = appendConversationMessage("assistant", answer, {
    pending: false,
    sources,
    runId: agentState.run_id || "",
  });
}

function renderAgentState(agentState, context) {
  if (!isCurrentAgentContext(context)) return;
  state.agentTimeline = (agentState.tool_trace || []).map((item) => ({
    type: "tool",
    call_id: item.call_id,
    tool_name: item.name,
    args: item.args,
    requires_approval: item.requires_approval,
    ok: item.ok,
    status: item.status,
    summary: item.result?.message || item.result?.error || item.status,
    result: item.result,
  }));
  renderAgentTimeline();
  renderAgentSources(agentState.sources || []);
  syncAgentAnswer(agentState);
  renderAgentApproval(agentState.pending_approval);
  // 空的新线程还没有 checkpoint 配置，保留工作台当前开关；已有运行状态以 checkpoint 为准。
  if (agentState.status !== "idle" || agentState.run_id || agentState.tool_call_count) {
    state.webSearchEnabled = Boolean(agentState.web_search_enabled);
  }
  syncWebSearchToggle();
  const status = agentState.pending_approval ? "等待审批：请在右侧审批面板处理。" : `状态：${agentState.status || "就绪"} · 工具 ${agentState.tool_call_count || 0} 次`;
  setAgentStatus(status);
  if (agentState.pending_approval) openContextPanel("approvalContext");
}

async function loadAgentState(threadId = state.agentThreadId, version = state.agentViewVersion) {
  const context = { threadId, version };
  state.agentStateAbortController?.abort();
  const controller = new AbortController();
  state.agentStateAbortController = controller;
  try {
    const result = await apiFetch(
      `/api/agent/threads/${encodeURIComponent(threadId)}/state`,
      { signal: controller.signal },
    );
    if (!isCurrentAgentContext(context)) return;
    renderAgentState(result, context);
  } catch (error) {
    if (error?.name === "AbortError" || !isCurrentAgentContext(context)) return;
    renderConversation();
    setAgentStatus(`状态加载失败：${getErrorMessage(error)}`);
  } finally {
    if (state.agentStateAbortController === controller) state.agentStateAbortController = null;
  }
}

function handleAgentEvent(event, context) {
  if (!isCurrentAgentContext(context)) return;
  if (event.event === "run_started") {
    upsertSessionMeta(context.threadId, { status: "running", updatedAt: Date.now() });
    setAgentStatus(event.resume ? "正在恢复上一次操作..." : "Agent 正在理解任务...");
    return;
  }
  if (event.event === "tool_call") {
    const existing = state.agentTimeline.find((item) => item.call_id === event.call_id);
    if (existing) Object.assign(existing, event, { type: "tool" });
    else state.agentTimeline.push({ type: "tool", ...event });
    renderAgentTimeline();
    upsertSessionMeta(context.threadId, { status: "running", updatedAt: Date.now() });
    setAgentStatus(`正在执行：${getToolLabel(event.tool_name)}`);
    openContextPanel("timelineContext");
    return;
  }
  if (event.event === "tool_result") {
    const existing = state.agentTimeline.find((item) => item.call_id === event.call_id);
    if (existing) Object.assign(existing, event, { type: "tool" });
    else state.agentTimeline.push({ type: "tool", ...event });
    renderAgentTimeline();
    upsertSessionMeta(context.threadId, { status: event.ok ? "running" : "failed", updatedAt: Date.now() });
    setAgentStatus(event.ok ? `已完成：${getToolLabel(event.tool_name)}` : `工具失败：${getToolLabel(event.tool_name)}`);
    return;
  }
  if (event.event === "approval_required") {
    renderAgentApproval(event);
    upsertSessionMeta(context.threadId, { status: "awaiting_approval", updatedAt: Date.now() });
    setAgentStatus("等待审批：批准或拒绝后才能继续。");
    openContextPanel("approvalContext");
    return;
  }
  if (event.event === "answer") {
    const answer = normalizeText(event.answer);
    if (answer) {
      const answerOptions = {
        pending: false,
        runId: event.run_id || "",
      };
      if (event.sources) answerOptions.sources = normalizeSources(event.sources);
      updateCurrentAssistant(answer, answerOptions);
      $("#agentAnswer").textContent = answer;
    }
    upsertSessionMeta(context.threadId, { status: "completed", updatedAt: Date.now() });
    if (event.sources) renderAgentSources(event.sources);
    setAgentStatus("回答已生成。");
    return;
  }
  if (event.event === "error") {
    const message = getErrorMessage(event.message, "执行失败");
    showNotice(`Agent：${message}`, "error");
    updateCurrentAssistant(message, { pending: false, meta: "执行失败" });
    upsertSessionMeta(context.threadId, { status: "failed", updatedAt: Date.now() });
    setAgentStatus("执行失败。");
    return;
  }
  if (event.event === "done") {
    setAgentBusy(false);
    if (event.sources) renderAgentSources(event.sources);
    if (event.status === "awaiting_approval") {
      upsertSessionMeta(context.threadId, { status: "awaiting_approval", updatedAt: Date.now() });
      setAgentStatus("等待审批：请在右侧审批面板处理。");
      openContextPanel("approvalContext");
    } else {
      upsertSessionMeta(context.threadId, { status: event.status === "completed" ? "completed" : "failed", updatedAt: Date.now() });
      renderAgentApproval(null);
      setAgentStatus(event.status === "completed" ? "本轮任务已完成。" : `任务状态：${event.status || "结束"}`);
    }
  }
}

async function runAgentRequest(input, threadId, version, resume = false, approval = null) {
  const context = { threadId, version };
  const controller = new AbortController();
  state.agentStreamAbortController = controller;
  try {
    const response = await fetch(
      resume
        ? `/api/agent/threads/${encodeURIComponent(threadId)}/resume/stream`
        : "/api/agent/runs/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(resume ? approval : {
          input,
          thread_id: threadId,
          web_search_enabled: state.webSearchEnabled,
        }),
      }
    );
    await consumeSse(response, (event) => handleAgentEvent(event, context));
    if (isCurrentAgentContext(context)) await loadAgentState(threadId, version);
  } finally {
    if (state.agentStreamAbortController === controller) state.agentStreamAbortController = null;
  }
}

function syncWebSearchToggle() {
  const toggle = $("#webSearchToggle");
  const text = $("#webSearchModeText");
  if (toggle) toggle.checked = state.webSearchEnabled;
  if (text) text.textContent = state.webSearchEnabled ? "开启：证据不足时搜索网页" : "关闭：仅使用知识库";
  try { localStorage.setItem("knowledge_agent_web_search", String(state.webSearchEnabled)); } catch { /* 忽略不可用的浏览器存储。 */ }
}

async function runAgent(event) {
  event.preventDefault();
  hideNotice();
  const input = $("#agentInput").value.trim();
  if (!input || state.agentBusy || state.agentPendingApproval) return;
  upsertSessionMeta(state.agentThreadId, { title: sessionTitleFromInput(input), status: "running", updatedAt: Date.now() });
  $("#agentInput").value = "";
  state.agentTimeline = [];
  state.agentPendingApproval = null;
  renderAgentTimeline();
  renderAgentApproval(null);
  appendConversationMessage("user", input);
  appendConversationMessage("assistant", "正在理解任务...", { pending: true });
  setAgentBusy(true);
  const threadId = state.agentThreadId;
  const version = state.agentViewVersion;
  try {
    await runAgentRequest(input, threadId, version);
  } catch (error) {
    if (!isCurrentAgentContext({ threadId, version }) || error?.name === "AbortError") return;
    const message = getErrorMessage(error);
    updateCurrentAssistant(message, { pending: false, meta: "请求失败" });
    showNotice(`Agent 请求失败：${message}`, "error");
    setAgentBusy(false);
  }
}

async function resumeAgent(decision) {
  const pending = state.agentPendingApproval;
  if (!pending || state.agentBusy) return;
  const reason = decision === "reject" ? window.prompt("请输入拒绝原因（可选）：", "") || "" : "";
  const button = decision === "approve" ? $("#approveAgentBtn") : $("#rejectAgentBtn");
  setLoading(button, true, decision === "approve" ? "批准中..." : "拒绝中...");
  state.agentBusy = true;
  const threadId = state.agentThreadId;
  const version = state.agentViewVersion;
  try {
    await runAgentRequest(null, threadId, version, true, {
      approval_id: pending.approval_id,
      decision,
      reason,
    });
  } catch (error) {
    if (!isCurrentAgentContext({ threadId, version }) || error?.name === "AbortError") return;
    showNotice(`审批提交失败：${getErrorMessage(error)}`, "error");
  } finally {
    state.agentBusy = false;
    setLoading(button, false);
  }
}

function newAgentThread() {
  const threadId = makeId("agent");
  upsertSessionMeta(threadId, { title: "新会话", status: "idle", updatedAt: Date.now() });
  activateAgentThread(threadId, { loadState: false });
  showNotice("已创建新会话。", "success");
}

async function deleteAgentThread(threadId = state.agentThreadId) {
  if (!threadId) return;
  const session = loadSessionHistory().find((item) => item.threadId === threadId);
  const title = session?.title || "当前会话";
  if (!window.confirm(`确认删除“${title}”？删除后无法恢复该 Agent 会话。`)) return;

  try {
    await apiFetch(`/api/agent/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" });
    const deletingCurrent = threadId === state.agentThreadId;
    removeSessionMeta(threadId);
    if (!deletingCurrent) {
      showNotice("会话已删除。", "success");
      return;
    }

    const next = loadSessionHistory()[0];
    if (next) {
      activateAgentThread(next.threadId);
    } else {
      const newThreadId = makeId("agent");
      upsertSessionMeta(newThreadId, { title: "新会话", status: "idle", updatedAt: Date.now() });
      activateAgentThread(newThreadId, { loadState: false });
    }
    showNotice("会话已删除。", "success");
  } catch (error) {
    showNotice(`删除会话失败：${getErrorMessage(error)}`, "error");
  }
}

function setDocumentView(view) {
  state.documentView = view === "batches" ? "batches" : "documents";
  try { localStorage.setItem("knowledge_agent_document_view", state.documentView); } catch { /* 忽略不可用的浏览器存储。 */ }
  document.querySelector(".document-overview")?.classList.toggle("show-batches", state.documentView === "batches");
  document.querySelectorAll("[data-document-view]").forEach((button) => {
    const active = button.dataset.documentView === state.documentView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll("[data-document-section]").forEach((section) => {
    const sectionType = section.dataset.documentSection;
    section.hidden = sectionType !== "danger" && sectionType !== state.documentView;
  });
}

function renderBatchItems(items = [], batchId = state.lastBatchId) {
  const tbody = $("#batchItemsTable");
  if (!tbody) return;
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">当前批次没有文件明细。</td></tr>`;
    return;
  }
  tbody.innerHTML = items.map((item) => {
    const identifier = item.document_id
      ? `Document ID: ${escapeHtml(item.document_id)}`
      : escapeHtml(item.error || "-");
    const retryButton = item.status === "failed"
      ? `<button class="button subtle retry-upload" type="button" data-batch-id="${escapeHtml(batchId)}" data-item-id="${escapeHtml(item.item_id)}">重试</button>`
      : "-";
    return `
      <tr>
        <td data-label="文件名">${escapeHtml(item.filename)}</td>
        <td data-label="状态">${formatUploadStatus(item.status)}</td>
        <td data-label="chunk">${escapeHtml(item.chunk_count)}</td>
        <td data-label="耗时">${escapeHtml(formatDuration(item.duration_ms))}</td>
        <td data-label="失败阶段">${escapeHtml(item.error_stage || "-")}</td>
        <td data-label="错误 / Document ID">${identifier}${item.error && item.document_id ? `<br><span class="error-text">${escapeHtml(item.error)}</span>` : ""}</td>
        <td data-label="操作">${retryButton}</td>
      </tr>
    `;
  }).join("");
}

function renderBatchResult(result) {
  state.lastBatchId = result.batch_id || state.lastBatchId;
  const summary = result.batch_id
    ? `批次 ${result.batch_id.slice(0, 12)}… · ${formatUploadStatus(result.status)} · 成功 ${result.succeeded} · 失败 ${result.failed} · 跳过 ${result.skipped}`
    : `${result.status === "duplicate" ? "重复文件已跳过" : "上传完成"} · ${result.filename || ""} · ${result.chunk_count || 0} 个 chunk`;
  $("#uploadResult").textContent = summary;
  $("#batchDetailLabel").textContent = result.batch_id ? `批次 ${result.batch_id.slice(0, 12)}…` : "单文件结果";
  renderBatchItems(result.items || [], state.lastBatchId);
  setDocumentView(result.batch_id ? "batches" : "documents");
}

function renderBatchHistoryList(batches = []) {
  const list = $("#batchHistoryList");
  if (!list) return;
  if (!batches.length) {
    list.innerHTML = `<div class="context-empty">暂无批次记录。</div>`;
    return;
  }
  list.innerHTML = batches.map((batch) => `
    <button class="batch-list-item batch-detail" type="button" data-batch-id="${escapeHtml(batch.batch_id)}">
      <span class="batch-list-main"><strong>${escapeHtml(batch.batch_id.slice(0, 12))}…</strong><span>成功 ${batch.succeeded} · 失败 ${batch.failed} · 跳过 ${batch.skipped} · ${escapeHtml(formatDuration(batch.duration_ms))}</span></span>
      ${formatUploadStatus(batch.status)}
    </button>
  `).join("");
}

async function loadBatchHistory() {
  const list = $("#batchHistoryList");
  if (list) list.innerHTML = `<div class="context-empty">加载中...</div>`;
  try {
    const batches = await apiFetch("/api/documents/upload-batches?limit=20");
    renderBatchHistoryList(batches);
  } catch (error) {
    if (list) list.innerHTML = `<div class="context-empty">加载失败：${escapeHtml(getErrorMessage(error))}</div>`;
  }
}

async function loadBatchDetail(batchId) {
  try {
    const result = await apiFetch(`/api/documents/upload-batches/${encodeURIComponent(batchId)}`);
    renderBatchResult(result);
    showNotice(`已加载批次：${result.status}`);
  } catch (error) {
    showNotice(`批次加载失败：${getErrorMessage(error)}`, "error");
  }
}

async function retryBatchItem(batchId, itemId, button) {
  setLoading(button, true, "重试中...");
  try {
    const result = await apiFetch(`/api/documents/upload-batches/${encodeURIComponent(batchId)}/items/${encodeURIComponent(itemId)}/retry`, { method: "POST" });
    renderBatchResult(result);
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
    showNotice(`重试完成：${result.status}`);
  } catch (error) {
    showNotice(`重试失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderUploadQueue(files = state.selectedFiles) {
  const queue = $("#uploadQueue");
  if (!queue) return;
  if (!files.length) {
    queue.innerHTML = `<span class="context-empty">尚未选择文件。</span>`;
    return;
  }
  queue.innerHTML = files.map((file, index) => {
    const error = validateUploadFile(file);
    return `
      <div class="upload-file-row ${error ? "has-error" : "is-valid"}">
        <span class="upload-file-state" aria-hidden="true"><svg class="icon"><use href="/static/icons.svg#${error ? "circle-help" : "file-text"}"></use></svg></span>
        <span class="upload-file-copy"><strong class="upload-file-name">${escapeHtml(file.name)}</strong><small>${formatFileSize(file.size)} · ${escapeHtml(error || "格式和大小检查通过")}</small></span>
        <button class="icon-button compact-icon-button remove-upload-file" type="button" data-file-index="${index}" title="移除 ${escapeHtml(file.name)}" aria-label="移除 ${escapeHtml(file.name)}"><svg class="icon"><use href="/static/icons.svg#x"></use></svg></button>
      </div>`;
  }).join("");
  const invalidCount = files.filter(validateUploadFile).length;
  const limit = $(".upload-limit");
  if (limit) limit.textContent = `${files.length}/${UPLOAD_MAX_FILES} 个文件 · ${invalidCount ? `${invalidCount} 个文件需要处理` : "格式和大小检查通过"}`;
}

async function uploadDocument(event) {
  event.preventDefault();
  hideNotice();
  const files = state.selectedFiles.slice();
  renderUploadQueue(files);
  if (!files.length) {
    showNotice("请先选择文件。", "error");
    return;
  }
  if (files.length > UPLOAD_MAX_FILES) {
    showNotice(`单批最多上传 ${UPLOAD_MAX_FILES} 个文件。`, "error");
    return;
  }
  const invalidFiles = files.filter(validateUploadFile);
  if (invalidFiles.length) {
    showNotice(`请先处理 ${invalidFiles.length} 个格式或大小不符合要求的文件。`, "error");
    return;
  }
  const formData = new FormData();
  for (const file of files) formData.append(files.length === 1 ? "file" : "files", file);
  const button = $("#uploadBtn");
  setLoading(button, true, files.length === 1 ? "上传中..." : "批量上传中...");
  try {
    const url = files.length === 1 ? "/api/documents/upload" : "/api/documents/upload-batch";
    const result = await apiFetch(url, { method: "POST", body: formData });
    renderBatchResult(result);
    const message = files.length === 1
      ? (result.status === "duplicate" ? `重复文件已跳过：${result.filename}。` : `上传成功：${result.filename}，生成 ${result.chunk_count} 个 chunk。`)
      : `批量上传完成：成功 ${result.succeeded} 个，失败 ${result.failed} 个，跳过 ${result.skipped} 个。`;
    showNotice(message);
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
    state.selectedFiles = [];
    $("#uploadFile").value = "";
    renderUploadQueue([]);
  } catch (error) {
    const message = getErrorMessage(error);
    $("#uploadResult").textContent = message;
    showNotice(`上传失败：${message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function clearVectorStore() {
  hideNotice();
  if (!window.confirm("确认清空知识库？文档、向量和历史记录都会被删除。")) return;
  const button = $("#clearVectorBtn");
  setLoading(button, true, "清理中...");
  try {
    const deleteFiles = $("#deleteFiles").checked;
    const result = await apiFetch(`/api/vector-store/clear?confirm=true&delete_files=${deleteFiles}`, { method: "DELETE" });
    $("#clearResult").textContent = `已清理 ${result.removed_documents} 个文档、${result.removed_chunks} 个 chunk。`;
    state.lastBatchId = null;
    renderBatchItems([]);
    await Promise.all([refreshStatus(), loadDocuments(), loadChunks(), loadBatchHistory()]);
    showNotice("知识库已清空。");
  } catch (error) {
    const message = getErrorMessage(error);
    $("#clearResult").textContent = message;
    showNotice(`清理失败：${message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function loadDocuments() {
  const tbody = $("#documentsTable");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="6" class="empty-state">加载中...</td></tr>`;
  try {
    const docs = await apiFetch("/api/documents");
    if (!docs.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-state">暂无文档。</td></tr>`;
      return;
    }
    tbody.innerHTML = docs.map((doc) => {
      const shortId = String(doc.document_id || "");
      const displayId = shortId.length > 20 ? `${shortId.slice(0, 8)}…${shortId.slice(-6)}` : shortId;
      return `
      <tr>
        <td data-label="文件名">${escapeHtml(doc.filename)}</td>
        <td data-label="类型">${escapeHtml(doc.file_type)}</td>
        <td data-label="切片数">${escapeHtml(doc.chunk_count)}</td>
        <td data-label="创建时间">${escapeHtml(doc.created_at)}</td>
        <td data-label="Document ID"><code class="document-id" title="${escapeHtml(doc.document_id)}">${escapeHtml(displayId)}</code><button class="icon-button compact-icon-button copy-document-id" type="button" data-document-id="${escapeHtml(doc.document_id)}" title="复制 Document ID" aria-label="复制 Document ID"><svg class="icon"><use href="/static/icons.svg#copy"></use></svg></button></td>
        <td data-label="操作" class="document-actions"><details class="document-actions-menu"><summary class="icon-button compact-icon-button" title="更多操作" aria-label="更多操作"><svg class="icon"><use href="/static/icons.svg#more-horizontal"></use></svg></summary><div class="document-menu"><button class="reindex-document" type="button" data-document-id="${escapeHtml(doc.document_id)}"><svg class="icon"><use href="/static/icons.svg#refresh-cw"></use></svg> 重建索引</button><button class="delete-document" type="button" data-document-id="${escapeHtml(doc.document_id)}"><svg class="icon"><use href="/static/icons.svg#trash-2"></use></svg> 删除文档</button></div></details></td>
      </tr>
    `;
    }).join("");
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">加载失败：${escapeHtml(getErrorMessage(error))}</td></tr>`;
  }
}

async function deleteDocument(documentId) {
  if (!window.confirm(`确认删除文档 ${documentId}？这会删除其向量、源文件和解析结果。`)) return;
  try {
    const result = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}?delete_files=true`, { method: "DELETE" });
    showNotice(`文档已删除：移除 ${result.removed_chunks} 个 chunk。`);
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
  } catch (error) {
    showNotice(`删除文档失败：${getErrorMessage(error)}`, "error");
  }
}

async function reindexDocument(documentId, button) {
  if (!window.confirm(`确认重新解析并索引文档 ${documentId}？`)) return;
  setLoading(button, true, "重建中...");
  try {
    const result = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}/reindex`, { method: "POST" });
    showNotice(`重建完成：${result.filename}，${result.chunk_count} 个 chunk。`);
    await Promise.all([refreshStatus(), loadDocuments()]);
  } catch (error) {
    showNotice(`重建索引失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

function getQuestionPayload() {
  const question = $("#questionInput").value.trim();
  if (!question) throw new Error("请输入问题。");
  return {
    question,
    thread_id: $("#threadId").value.trim() || "default",
    top_k: parseIntInput("#qaTopK", 3),
    retrieval_strategy: $("#retrievalStrategy").value,
    document_id: $("#documentIdFilter").value.trim() || null,
    filename: $("#filenameFilter").value.trim() || null,
  };
}

function renderSourcesBox(sources = []) {
  renderSources("#sourcesBox", sources);
}

function setExperimentAnswer(answer, sources = []) {
  $("#answerText").innerHTML = markdownToHtml(answer, sources);
  renderSourcesBox(sources);
}

async function askRag() {
  hideNotice();
  const button = $("#askBtn");
  setLoading(button, true, "运行中...");
  try {
    const result = await apiFetch("/api/rag/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(getQuestionPayload()) });
    setExperimentAnswer(result.answer, result.sources);
  } catch (error) {
    showNotice(`问答失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function streamRag() {
  hideNotice();
  const button = $("#askBtn");
  setLoading(button, true, "流式中...");
  $("#answerText").textContent = "";
  $("#sourcesBox").innerHTML = "";
  try {
    const response = await fetch("/api/rag/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(getQuestionPayload()) });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let answer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) {
        const line = event.split("\n").find((item) => item.startsWith("data: "));
        if (!line || line.slice(6) === "[DONE]") continue;
        answer += line.slice(6);
        $("#answerText").innerHTML = markdownToHtml(answer);
      }
    }
  } catch (error) {
    showNotice(`流式问答失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

function renderDynamicExtra(extra = {}, answer = "") {
  const chunks = extra.retrieved_chunks || [];
  setExperimentAnswer(answer, chunks.map((chunk) => ({
    filename: chunk.metadata?.filename || "unknown",
    chunk_index: chunk.metadata?.chunk_index ?? "-",
    score: chunk.score ?? "-",
    content_preview: chunk.content || "",
  })));
}

async function askDynamicRag() {
  hideNotice();
  const button = $("#askBtn");
  setLoading(button, true, "Dynamic RAG 中...");
  try {
    const question = $("#questionInput").value.trim();
    if (!question) throw new Error("请输入问题。");
    const result = await apiFetch("/api/agent/dynamic-rag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input: question, thread_id: $("#threadId").value.trim() || "default" }) });
    renderDynamicExtra(result.extra || {}, result.output || "");
  } catch (error) {
    showNotice(`Dynamic RAG 失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function askLangGraphAgent() {
  hideNotice();
  const button = $("#askBtn");
  setLoading(button, true, "Agent 中...");
  try {
    const question = $("#questionInput").value.trim();
    if (!question) throw new Error("请输入问题。");
    const result = await apiFetch("/api/agent/invoke", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input: question, thread_id: $("#threadId").value.trim() || "default" }) });
    setExperimentAnswer(result.output || "", result.extra?.sources || result.sources || []);
  } catch (error) {
    showNotice(`LangGraph Agent 失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function runSelectedExperiment() {
  if (state.retrievalMode === "dynamic") return askDynamicRag();
  if (state.experimentMode === "stream") return streamRag();
  if (state.experimentMode === "agent") return askLangGraphAgent();
  return askRag();
}

async function loadChunks() {
  const limit = parseIntInput("#chunkLimit", 20);
  const offset = parseIntInput("#chunkOffset", 0);
  const list = $("#chunkList");
  $("#chunksSummary").textContent = "加载中...";
  list.innerHTML = "";
  try {
    const result = await apiFetch(`/api/vector-store/chunks?limit=${limit}&offset=${offset}`);
    $("#chunksSummary").textContent = `共 ${result.total} 个 chunk，当前显示 ${result.chunks.length} 个。`;
    if (!result.chunks.length) {
      list.innerHTML = `<div class="empty-state">没有 chunk。</div>`;
      return;
    }
    list.innerHTML = result.chunks.map((chunk) => `
      <details class="chunk-card"><summary class="chunk-summary"><strong>${escapeHtml(chunk.filename || "未知文档")}</strong><span class="chunk-meta">切片 ${escapeHtml(chunk.chunk_index)} · ${escapeHtml(chunk.vector_dimension)} 维</span></summary><pre>${escapeHtml(chunk.content)}</pre></details>
    `).join("");
  } catch (error) {
    $("#chunksSummary").textContent = `加载失败：${getErrorMessage(error)}`;
  }
}

const RAGAS_DEFAULT_PATHS = {
  baseline: { samples: "eval_outputs/ragas_samples_baseline.json", result: "eval_outputs/ragas_result_baseline.json" },
  hyde_rewrite: { samples: "eval_outputs/ragas_samples_hyde_rewrite.json", result: "eval_outputs/ragas_result_hyde_rewrite.json" },
};

function getRagasDefaultPaths(mode, strategy) {
  if (strategy === "dense") return RAGAS_DEFAULT_PATHS[mode];
  return { samples: `eval_outputs/ragas_samples_${mode}_${strategy}.json`, result: `eval_outputs/ragas_result_${mode}_${strategy}.json` };
}

function syncRagasOutputPaths() {
  const mode = $("#retrievalMode").value;
  const strategy = $("#ragasRetrievalStrategy").value;
  const previousPaths = getRagasDefaultPaths(lastRagasMode, lastRagasStrategy);
  const nextPaths = getRagasDefaultPaths(mode, strategy);
  if (!$("#samplesOutputPath").value || $("#samplesOutputPath").value === previousPaths.samples) $("#samplesOutputPath").value = nextPaths.samples;
  if (!$("#resultOutputPath").value || $("#resultOutputPath").value === previousPaths.result) $("#resultOutputPath").value = nextPaths.result;
  lastRagasMode = mode;
  lastRagasStrategy = strategy;
}

function getRagasPayload() {
  const mode = $("#retrievalMode").value;
  const strategy = $("#ragasRetrievalStrategy").value;
  const defaults = getRagasDefaultPaths(mode, strategy);
  return {
    dataset_path: $("#datasetPath").value.trim() || "eval_data/ai_agents_in_depth_eval_dataset.json",
    retrieval_mode: mode,
    retrieval_strategy: strategy,
    samples_output_path: $("#samplesOutputPath").value.trim() || defaults.samples,
    result_output_path: $("#resultOutputPath").value.trim() || defaults.result,
    top_k: parseIntInput("#ragasTopK", 3),
    max_samples: parseOptionalInt("#maxSamples"),
    force_rebuild: $("#forceRebuild").checked,
  };
}

function renderRagasResult(result) {
  $("#ragasResult").textContent = JSON.stringify({ sample_count: result.sample_count, retrieval_mode: result.retrieval_mode, retrieval_strategy: result.retrieval_strategy, cache_used: result.cache_used, cache_status: result.cache_status, samples_path: result.samples_path, result_path: result.result_path, csv_path: result.csv_path, top_k: result.top_k, max_samples: result.max_samples, retrieval_metrics: result.retrieval_metrics, metric_null_counts: result.metric_null_counts, summary_path: result.summary_path }, null, 2);
  const metrics = { ...(result.metrics || {}), ...(result.retrieval_metrics || {}) };
  $("#metricsGrid").innerHTML = Object.entries(metrics).map(([name, value]) => `<div class="metric-card"><span>${escapeHtml(formatRagasMetricName(name))}</span><strong>${escapeHtml(value == null ? "-" : value)}</strong></div>`).join("");
}

async function buildRagasSamples() {
  hideNotice();
  const button = $("#buildSamplesBtn");
  setLoading(button, true, "生成中...");
  try {
    const result = await apiFetch("/api/evaluation/ragas/build-samples", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(getRagasPayload()) });
    renderRagasResult(result);
    showNotice(`样本已生成：${result.sample_count} 条。`);
  } catch (error) {
    showNotice(`生成样本失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function runRagas() {
  hideNotice();
  if (!window.confirm("RAGAS 完整评估可能较慢并消耗 DeepSeek token，确认继续？")) return;
  const button = $("#runRagasBtn");
  setLoading(button, true, "评估中...");
  try {
    const result = await apiFetch("/api/evaluation/ragas/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(getRagasPayload()) });
    renderRagasResult(result);
    showNotice(`评估完成：${result.sample_count} 条。`);
  } catch (error) {
    showNotice(`评估失败：${getErrorMessage(error)}`, "error");
  } finally {
    setLoading(button, false);
  }
}

function showMobileMoreMenu() {
  let menu = $("#mobileMoreMenu");
  if (menu) {
    menu.remove();
    return;
  }
  menu = document.createElement("div");
  menu.id = "mobileMoreMenu";
  menu.className = "mobile-more-menu";
  const sessions = loadSessionHistory().slice(0, 5);
  const sessionItems = sessions.length
    ? sessions.map((session) => `
      <div class="mobile-session-row">
        <button type="button" class="mobile-session-item" data-session-thread="${escapeHtml(session.threadId)}" title="${escapeHtml(session.title)}">
          <span class="session-status-dot status-${escapeHtml(session.status || "idle")}" aria-hidden="true"></span><span>${escapeHtml(session.title)}</span>
        </button>
        <button type="button" class="mobile-session-delete" data-session-delete="${escapeHtml(session.threadId)}" title="删除会话" aria-label="删除会话 ${escapeHtml(session.title)}">
          <svg class="icon" aria-hidden="true"><use href="/static/icons.svg#trash-2"></use></svg>
        </button>
      </div>
    `).join("")
    : `<span class="mobile-more-empty">暂无历史会话</span>`;
  menu.innerHTML = `<button type="button" data-menu-tab="chunks"><svg class="icon"><use href="/static/icons.svg#search"></use></svg> 检索调试</button><button type="button" data-menu-tab="ragas"><svg class="icon"><use href="/static/icons.svg#chart-no-axes-combined"></use></svg> RAGAS 评估</button><div class="mobile-more-heading"><svg class="icon"><use href="/static/icons.svg#messages-square"></use></svg> 最近会话</div>${sessionItems}<a href="/docs" target="_blank" rel="noreferrer"><svg class="icon"><use href="/static/icons.svg#external-link"></use></svg> API 文档</a>`;
  document.body.append(menu);
  menu.querySelectorAll("[data-menu-tab]").forEach((button) => button.addEventListener("click", () => {
    menu.remove();
    switchTab(button.dataset.menuTab);
  }));
  menu.querySelectorAll("[data-session-thread]").forEach((button) => button.addEventListener("click", () => {
    menu.remove();
    switchTab("agent");
    selectAgentThread(button.dataset.sessionThread);
  }));
}

function switchTab(tabName) {
  if (tabName === "more") {
    showMobileMoreMenu();
    return;
  }
  if (tabName !== "agent" && isContextOpen()) closeContextPanel({ restoreFocus: false });
  state.currentTab = tabName;
  document.body.classList.toggle("agent-view", tabName === "agent");
  $$(".tab-button").forEach((button) => {
    if (!button.dataset.tab || button.dataset.tab === "more") return;
    const active = button.dataset.tab === tabName;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `${tabName}Panel`));
  if (tabName === "documents") {
    loadDocuments();
    loadBatchHistory();
  }
  if (tabName === "chunks") loadChunks();
  if (tabName === "agent") loadAgentState();
}

function bindContextTabs() {
  $$(".context-tab").forEach((button) => button.addEventListener("click", () => selectContextView(button.dataset.context)));
  $$(".context-rail-button").forEach((button) => button.addEventListener("click", () => openContextPanel(button.dataset.context)));
}

function bindTabListKeyboard() {
  $$("[role=tablist]").forEach((tabList) => {
    const tabs = () => Array.from(tabList.querySelectorAll("[role=tab]"));
    tabList.addEventListener("keydown", (event) => {
      const supportedKeys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
      if (!supportedKeys.includes(event.key)) return;
      const items = tabs();
      const current = items.indexOf(document.activeElement);
      if (current < 0) return;
      event.preventDefault();
      const nextIndex = event.key === "Home"
        ? 0
        : event.key === "End"
          ? items.length - 1
          : (current + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      const next = items[nextIndex];
      next.focus();
      if (next.dataset.context) selectContextView(next.dataset.context);
      if (next.dataset.documentView) setDocumentView(next.dataset.documentView);
    });
  });
}

function bindUploadDropzone() {
  const input = $("#uploadFile");
  const zone = $(".upload-dropzone");
  if (!input || !zone) return;
  input.addEventListener("change", () => {
    state.selectedFiles = Array.from(input.files || []);
    renderUploadQueue();
  });
  ["dragenter", "dragover"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.add("drag-over");
  }));
  ["dragleave", "drop"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.remove("drag-over");
  }));
  zone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer?.files || []);
    if (!files.length) return;
    state.selectedFiles = files;
    renderUploadQueue();
    try {
      const transfer = new DataTransfer();
      files.forEach((file) => transfer.items.add(file));
      input.files = transfer.files;
    } catch {
      // 某些浏览器不允许脚本设置 input.files，队列仍会提供反馈。
    }
  });
}

function bindEvents() {
  $$(".tab-button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  bindContextTabs();
  bindTabListKeyboard();
  bindUploadDropzone();
  document.querySelectorAll("[data-document-view]").forEach((button) => button.addEventListener("click", () => setDocumentView(button.dataset.documentView)));
  $("#uploadForm")?.addEventListener("submit", uploadDocument);
  $("#agentForm")?.addEventListener("submit", runAgent);
  $("#agentInput")?.addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
  });
  $("#agentInput")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#agentForm").requestSubmit();
    }
  });
  $("#webSearchToggle")?.addEventListener("change", (event) => {
    state.webSearchEnabled = Boolean(event.target.checked);
    syncWebSearchToggle();
    setAgentStatus(state.webSearchEnabled ? "已允许联网兜底；仍会优先检索知识库。" : "已关闭联网搜索，仅依据知识库回答。");
  });
  $$(".suggestion-chip").forEach((button) => button.addEventListener("click", () => {
    $("#agentInput").value = button.dataset.suggestion || "";
    $("#agentInput").dispatchEvent(new Event("input"));
    $("#agentInput").focus();
  }));
  $("#retrievalMode")?.addEventListener("change", syncRagasOutputPaths);
  $("#ragasRetrievalStrategy")?.addEventListener("change", syncRagasOutputPaths);
  onClick("#refreshDocumentsBtn", () => { refreshStatus(); loadDocuments(); loadBatchHistory(); });
  onClick("#clearVectorBtn", clearVectorStore);
  onClick("#refreshBatchHistoryBtn", loadBatchHistory);
  onClick("#askBtn", runSelectedExperiment);
  onClick("#streamBtn", streamRag);
  onClick("#langGraphBtn", askLangGraphAgent);
  onClick("#dynamicBtn", askDynamicRag);
  onClick("#clearAnswerBtn", () => { $("#answerText").textContent = "还没有回答。"; $("#sourcesBox").innerHTML = ""; });
  onClick("#loadChunksBtn", loadChunks);
  onClick("#buildSamplesBtn", buildRagasSamples);
  onClick("#runRagasBtn", runRagas);
  onClick("#refreshAgentStateBtn", () => loadAgentState());
  onClick("#newThreadBtn", newAgentThread);
  onClick("#approveAgentBtn", () => resumeAgent("approve"));
  onClick("#rejectAgentBtn", () => resumeAgent("reject"));
  onClick("#clearAgentBtn", () => {
    deleteAgentThread();
  });
  onClick("#openContextBtn", () => openContextPanel());
  onClick("#closeContextBtn", () => closeContextPanel());
  onClick("#contextScrim", () => closeContextPanel({ restoreFocus: false }));
  onClick("#newSessionSideBtn", newAgentThread);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isContextOpen()) closeContextPanel();
    trapContextFocus(event);
  });

  document.addEventListener("click", (event) => {
    const retryButton = event.target.closest(".retry-upload");
    if (retryButton) return retryBatchItem(retryButton.dataset.batchId, retryButton.dataset.itemId, retryButton);
    const detailButton = event.target.closest(".batch-detail");
    if (detailButton) return loadBatchDetail(detailButton.dataset.batchId);
    const deleteButton = event.target.closest(".delete-document");
    if (deleteButton) return deleteDocument(deleteButton.dataset.documentId);
    const reindexButton = event.target.closest(".reindex-document");
    if (reindexButton) return reindexDocument(reindexButton.dataset.documentId, reindexButton);
    const sessionDeleteButton = event.target.closest("[data-session-delete]");
    if (sessionDeleteButton) {
      $("#mobileMoreMenu")?.remove();
      return deleteAgentThread(sessionDeleteButton.dataset.sessionDelete);
    }
    const sessionButton = event.target.closest(".session-item");
    if (sessionButton) return selectAgentThread(sessionButton.dataset.threadId);
    const removeFileButton = event.target.closest(".remove-upload-file");
    if (removeFileButton) {
      const index = Number(removeFileButton.dataset.fileIndex);
      state.selectedFiles.splice(index, 1);
      renderUploadQueue();
      const input = $("#uploadFile");
      try {
        const transfer = new DataTransfer();
        state.selectedFiles.forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
      } catch { /* 队列状态仍然有效，下一次选择文件会刷新 input。 */ }
      return;
    }
    const copyButton = event.target.closest(".copy-document-id");
    if (copyButton) {
      navigator.clipboard?.writeText(copyButton.dataset.documentId || "").then(
        () => showNotice("Document ID 已复制。"),
        () => showNotice("复制失败，请手动选择 ID。", "error")
      );
      return;
    }
    const sourceRef = event.target.closest(".source-ref");
    if (sourceRef) {
      const index = Number(sourceRef.dataset.sourceIndex);
      const messageIndex = Number(sourceRef.dataset.messageIndex);
      const messageSources = Number.isInteger(messageIndex)
        ? normalizeSources(loadConversation()[messageIndex]?.sources)
        : state.agentSources;
      if (messageSources.length) renderAgentSources(messageSources);
      openContextPanel("sourcesContext");
      const card = $(`#source-card-${index}`);
      if (card) {
        card.open = true;
        card.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  });
}

function bindExperimentModes() {
  const labels = { standard: "普通 RAG", stream: "流式 RAG", agent: "LangGraph Agent", dynamic: "Dynamic RAG" };
  $$(".answer-mode-control .segment").forEach((button) => button.addEventListener("click", () => {
    state.experimentMode = button.dataset.mode;
    $$(".answer-mode-control .segment").forEach((item) => item.classList.toggle("active", item === button));
    $("#experimentModeLabel").textContent = `当前：${labels[state.experimentMode]}`;
  }));
  $$(".retrieval-mode-control .segment").forEach((button) => button.addEventListener("click", () => {
    state.retrievalMode = button.dataset.retrievalMode;
    $$(".retrieval-mode-control .segment").forEach((item) => item.classList.toggle("active", item === button));
    const dynamic = state.retrievalMode === "dynamic";
    $("#experimentModeLabel").textContent = `${labels[state.experimentMode]} · ${dynamic ? "Query Rewrite + HyDE" : "原问题检索"}`;
  }));
}

async function init() {
  initAgentThread();
  syncWebSearchToggle();
  document.body.classList.add("agent-view");
  renderConversation();
  bindEvents();
  bindExperimentModes();
  renderUploadQueue([]);
  setDocumentView("documents");
  await Promise.all([refreshStatus(), loadAgentState(), loadDocuments(), loadBatchHistory()]);
}

init();
