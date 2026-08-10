const state = {
  currentTab: "documents",
  lastBatchId: null,
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

function showNotice(message, type = "success") {
  const notice = $("#notice");
  notice.hidden = false;
  notice.className = `notice ${type}`;
  notice.textContent = message;
}

function hideNotice() {
  $("#notice").hidden = true;
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
    const detail = typeof payload === "object" ? payload.detail || JSON.stringify(payload) : payload;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}

function parseIntInput(selector, fallback) {
  const value = $(selector).value;
  if (value === "") return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOptionalInt(selector) {
  const value = $(selector).value.trim();
  if (!value) return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

async function refreshStatus() {
  try {
    const health = await apiFetch("/api/health");
    const status = await apiFetch("/api/status");
    $("#healthText").textContent = health.status || "ok";
    $("#documentCount").textContent = status.document_count;
    $("#chunkCount").textContent = status.chunk_count;
    $("#embeddingProvider").textContent = status.embedding_provider;
  } catch (error) {
    $("#healthText").textContent = "异常";
    showNotice(`状态加载失败：${error.message}`, "error");
  }
}

async function loadDocuments() {
  const tbody = $("#documentsTable");
  tbody.innerHTML = `<tr><td colspan="6">加载中...</td></tr>`;
  try {
    const docs = await apiFetch("/api/documents");
    if (!docs.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-state">暂无文档。</td></tr>`;
      return;
    }

    tbody.innerHTML = docs
      .map(
        (doc) => `
          <tr>
            <td>${escapeHtml(doc.filename)}</td>
            <td>${escapeHtml(doc.file_type)}</td>
            <td>${escapeHtml(doc.chunk_count)}</td>
            <td>${escapeHtml(doc.created_at)}</td>
            <td>${escapeHtml(doc.document_id)}</td>
            <td class="document-actions">
              <button class="button subtle reindex-document" type="button" data-document-id="${escapeHtml(doc.document_id)}">重建索引</button>
              <button class="button danger-text delete-document" type="button" data-document-id="${escapeHtml(doc.document_id)}">删除</button>
            </td>
          </tr>
        `
      )
      .join("");
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">加载失败：${escapeHtml(error.message)}</td></tr>`;
  }
}

async function deleteDocument(documentId) {
  const ok = window.confirm(`确认删除文档 ${documentId}？这会删除其向量、源文件和解析结果。`);
  if (!ok) return;
  try {
    const result = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}?delete_files=true`, {
      method: "DELETE",
    });
    showNotice(`文档已删除：移除 ${result.removed_chunks} 个 chunk。`);
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
  } catch (error) {
    showNotice(`删除文档失败：${error.message}`, "error");
  }
}

async function reindexDocument(documentId, button) {
  const ok = window.confirm(`确认重新解析并索引文档 ${documentId}？`);
  if (!ok) return;
  setLoading(button, true, "重建中...");
  try {
    const result = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}/reindex`, {
      method: "POST",
    });
    showNotice(`重建完成：${result.filename}，${result.chunk_count} 个 chunk。`);
    await Promise.all([refreshStatus(), loadDocuments()]);
  } catch (error) {
    showNotice(`重建索引失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

const UPLOAD_STATUS_LABELS = {
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

function formatUploadStatus(status) {
  const label = UPLOAD_STATUS_LABELS[status] || status || "未知";
  return `<span class="status-badge status-${escapeHtml(status || "unknown")}">${escapeHtml(label)}</span>`;
}

function formatDuration(durationMs) {
  const value = Number(durationMs || 0);
  return value ? `${value} ms` : "-";
}

function renderBatchItems(items = [], batchId = state.lastBatchId) {
  const tbody = $("#batchItemsTable");
  if (!tbody) return;
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">当前批次没有文件明细。</td></tr>`;
    return;
  }

  tbody.innerHTML = items
    .map((item) => {
      const identifier = item.document_id
        ? `Document ID: ${escapeHtml(item.document_id)}`
        : escapeHtml(item.error || "-");
      const retryButton = item.status === "failed"
        ? `<button class="button subtle retry-upload" type="button" data-batch-id="${escapeHtml(batchId)}" data-item-id="${escapeHtml(item.item_id)}">重试</button>`
        : "-";
      return `
        <tr>
          <td>${escapeHtml(item.filename)}</td>
          <td>${formatUploadStatus(item.status)}</td>
          <td>${escapeHtml(item.chunk_count)}</td>
          <td>${escapeHtml(formatDuration(item.duration_ms))}</td>
          <td>${escapeHtml(item.error_stage || "-")}</td>
          <td>${identifier}${item.error && item.document_id ? `<br><span class="error-text">${escapeHtml(item.error)}</span>` : ""}</td>
          <td>${retryButton}</td>
        </tr>
      `;
    })
    .join("");
}

function renderBatchResult(result) {
  state.lastBatchId = result.batch_id || state.lastBatchId;
  $("#uploadResult").textContent = JSON.stringify(result, null, 2);
  renderBatchItems(result.items || [], state.lastBatchId);
}

async function loadBatchHistory() {
  const tbody = $("#batchHistoryTable");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="6">加载中...</td></tr>`;
  try {
    const batches = await apiFetch("/api/documents/upload-batches?limit=20");
    if (!batches.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-state">暂无批次记录。</td></tr>`;
      return;
    }
    tbody.innerHTML = batches
      .map(
        (batch) => `
          <tr>
            <td title="${escapeHtml(batch.batch_id)}">${escapeHtml(batch.batch_id.slice(0, 12))}...</td>
            <td>${formatUploadStatus(batch.status)}</td>
            <td>成功 ${escapeHtml(batch.succeeded)} / 失败 ${escapeHtml(batch.failed)} / 跳过 ${escapeHtml(batch.skipped)} / 共 ${escapeHtml(batch.total)}</td>
            <td>${escapeHtml(formatDuration(batch.duration_ms))}</td>
            <td>${escapeHtml(batch.created_at)}</td>
            <td><button class="button subtle batch-detail" type="button" data-batch-id="${escapeHtml(batch.batch_id)}">查看</button></td>
          </tr>
        `
      )
      .join("");
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">加载失败：${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadBatchDetail(batchId) {
  try {
    const result = await apiFetch(`/api/documents/upload-batches/${encodeURIComponent(batchId)}`);
    renderBatchResult(result);
    showNotice(`已加载批次：${result.status}`);
  } catch (error) {
    showNotice(`批次加载失败：${error.message}`, "error");
  }
}

async function retryBatchItem(batchId, itemId, button) {
  setLoading(button, true, "重试中...");
  try {
    const result = await apiFetch(
      `/api/documents/upload-batches/${encodeURIComponent(batchId)}/items/${encodeURIComponent(itemId)}/retry`,
      { method: "POST" }
    );
    renderBatchResult(result);
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
    showNotice(`重试完成：${result.status}`);
  } catch (error) {
    showNotice(`重试失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  hideNotice();
  const files = Array.from($("#uploadFile").files || []);
  if (!files.length) {
    showNotice("请先选择文件。", "error");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append(files.length === 1 ? "file" : "files", file);
  }
  const button = $("#uploadBtn");
  setLoading(button, true, files.length === 1 ? "上传中..." : "批量上传中...");

  try {
    const url = files.length === 1 ? "/api/documents/upload" : "/api/documents/upload-batch";
    showNotice(files.length === 1 ? "正在调用单文件上传接口..." : `正在调用批量上传接口，共 ${files.length} 个文件...`);
    const result = await apiFetch(url, {
      method: "POST",
      body: formData,
    });
    if (files.length > 1) {
      renderBatchResult(result);
    } else {
      $("#uploadResult").textContent = JSON.stringify(result, null, 2);
      renderBatchItems([], state.lastBatchId);
    }
    if (files.length === 1) {
      const message = result.status === "duplicate"
        ? `重复文件已跳过：${result.filename}。`
        : `上传成功：${result.filename}，生成 ${result.chunk_count} 个 chunk。`;
      showNotice(message);
    } else {
      showNotice(`批量上传完成：成功 ${result.succeeded} 个，失败 ${result.failed} 个，跳过 ${result.skipped} 个。`);
    }
    await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
  } catch (error) {
    $("#uploadResult").textContent = error.message;
    showNotice(`上传失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function clearVectorStore() {
  hideNotice();
  const ok = window.confirm("确认清空向量库、文档元数据和对话历史？");
  if (!ok) return;

  const button = $("#clearVectorBtn");
  setLoading(button, true, "清理中...");
  const deleteFiles = $("#deleteFiles").checked;

  try {
    const result = await apiFetch(`/api/vector-store/clear?confirm=true&delete_files=${deleteFiles}`, {
      method: "DELETE",
    });
    $("#clearResult").textContent = JSON.stringify(result, null, 2);
    showNotice("向量库已清空。");
    state.lastBatchId = null;
    renderBatchItems([]);
    await Promise.all([refreshStatus(), loadDocuments(), loadChunks(), loadBatchHistory()]);
  } catch (error) {
    $("#clearResult").textContent = error.message;
    showNotice(`清理失败：${error.message}`, "error");
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

function renderSources(sources = []) {
  const box = $("#sourcesBox");
  if (!sources.length) {
    box.innerHTML = "";
    return;
  }

  box.innerHTML = sources
    .map(
      (source, index) => `
        <details class="source-card">
          <summary class="source-summary">
            <strong>${index + 1}. ${escapeHtml(source.filename || "unknown")}</strong>
            <span class="source-meta">${escapeHtml(source.retrieval_strategy || "dense")} / 切片 ${escapeHtml(source.chunk_index)} / 得分 ${escapeHtml(source.score)}${source.bm25_rank ? ` / BM25#${escapeHtml(source.bm25_rank)}` : ""}</span>
          </summary>
          <pre>${escapeHtml(source.content_preview)}</pre>
        </details>
      `
    )
    .join("");
}

function renderDynamicExtra(extra = {}) {
  const chunks = extra.retrieved_chunks || [];
  const graphPath = (extra.graph_path || []).join(" -> ");
  const queryInfo = `
Dynamic RAG 调试信息

Graph 全程路径 (graph_path):
${graphPath}

原始问题 (original_query):
${extra.original_query || ""}

问题重写 (rewritten_query):
${extra.rewritten_query || ""}

HyDE 虚构答案 (hyde_answer):
${extra.hyde_answer || ""}

实际检索文本 (retrieval_query):
${extra.retrieval_query || ""}

上下文是否足够 (context_sufficient):
${extra.context_sufficient ? "是" : "否"}

上下文充分性判断原因 (context_evaluation_reason):
${extra.context_evaluation_reason || ""}
`;
  $("#answerText").textContent = queryInfo;

  const box = $("#sourcesBox");
  if (!chunks.length) {
    box.innerHTML = `<div class="empty-state">没有返回 retrieved_chunks。</div>`;
    return;
  }

  box.innerHTML = chunks
    .map((chunk) => {
      const meta = chunk.metadata || {};
      const filename = meta.filename || "unknown";
      const chunkIndex = meta.chunk_index ?? "-";
      const score = chunk.score ?? "-";
      return `
        <details class="chunk-card">
          <summary class="chunk-summary">
            <strong>#${escapeHtml(chunk.rank)} ${escapeHtml(filename)}</strong>
            <span class="chunk-meta">切片 ${escapeHtml(chunkIndex)} / 得分 ${escapeHtml(score)}</span>
          </summary>
          <pre>${escapeHtml(chunk.content)}</pre>
        </details>
      `;
    })
    .join("");
}

async function askRag() {
  hideNotice();
  const button = $("#askBtn");
  setLoading(button, true, "问答中...");
  try {
    const payload = getQuestionPayload();
    const result = await apiFetch("/api/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#answerText").textContent = result.answer;
    renderSources(result.sources);
  } catch (error) {
    showNotice(`问答失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function streamRag() {
  hideNotice();
  const button = $("#streamBtn");
  setLoading(button, true, "流式中...");
  $("#answerText").textContent = "";
  $("#sourcesBox").innerHTML = "";

  try {
    const payload = getQuestionPayload();
    const response = await fetch("/api/rag/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        const line = event.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        $("#answerText").textContent += data;
      }
    }
  } catch (error) {
    showNotice(`流式问答失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function askDynamicRag() {
  hideNotice();
  const button = $("#dynamicBtn");
  setLoading(button, true, "Dynamic RAG 中...");
  try {
    const question = $("#questionInput").value.trim();
    if (!question) throw new Error("请输入问题。");
    const result = await apiFetch("/api/agent/dynamic-rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: question,
        thread_id: $("#threadId").value.trim() || "default",
      }),
    });
    const answer = `模型回答：\n${result.output}\n\n`;
    renderDynamicExtra(result.extra);
    $("#answerText").textContent = answer + $("#answerText").textContent;
  } catch (error) {
    showNotice(`Dynamic RAG 失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function askLangGraphAgent() {
  hideNotice();
  const button = $("#langGraphBtn");
  setLoading(button, true, "LangGraph Agent 中...");
  $("#sourcesBox").innerHTML = "";
  try {
    const question = $("#questionInput").value.trim();
    if (!question) throw new Error("请输入问题。");
    const result = await apiFetch("/api/agent/invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: question,
        thread_id: $("#threadId").value.trim() || "default",
      }),
    });
    const extra = result.extra || {};
    const graphPath = (extra.graph_path || []).join(" -> ");
    $("#answerText").textContent = `LangGraph Agent 调试信息

Graph 全程路径 (graph_path):
${graphPath}

路由结果 (route):
${extra.route || ""}

路由原因 (route_reason):
${extra.route_reason || ""}

实际工具 (tool_used):
${result.tool_used || ""}

工具输出 (output):
${result.output || ""}`;
  } catch (error) {
    showNotice(`LangGraph Agent 失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
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

    list.innerHTML = result.chunks
      .map(
        (chunk) => `
          <details class="chunk-card">
            <summary class="chunk-summary">
              <strong>${escapeHtml(chunk.filename || "unknown")}</strong>
              <span class="chunk-meta">
                切片 ${escapeHtml(chunk.chunk_index)} / ${escapeHtml(chunk.content_length)} 字符 / ${escapeHtml(chunk.vector_dimension)} 维
              </span>
            </summary>
            <pre>${escapeHtml(chunk.content)}</pre>
          </details>
        `
      )
      .join("");
  } catch (error) {
    $("#chunksSummary").textContent = `加载失败：${error.message}`;
  }
}

const RAGAS_DEFAULT_PATHS = {
  baseline: {
    samples: "eval_outputs/ragas_samples_baseline.json",
    result: "eval_outputs/ragas_result_baseline.json",
  },
  hyde_rewrite: {
    samples: "eval_outputs/ragas_samples_hyde_rewrite.json",
    result: "eval_outputs/ragas_result_hyde_rewrite.json",
  },
};

let lastRagasMode = "baseline";
let lastRagasStrategy = "dense";

function getRagasDefaultPaths(mode, strategy) {
  if (strategy === "dense") return RAGAS_DEFAULT_PATHS[mode];
  return {
    samples: `eval_outputs/ragas_samples_${mode}_${strategy}.json`,
    result: `eval_outputs/ragas_result_${mode}_${strategy}.json`,
  };
}

function syncRagasOutputPaths() {
  const mode = $("#retrievalMode").value;
  const strategy = $("#ragasRetrievalStrategy").value;
  const previousPaths = getRagasDefaultPaths(lastRagasMode, lastRagasStrategy);
  const nextPaths = getRagasDefaultPaths(mode, strategy);
  const samplesInput = $("#samplesOutputPath");
  const resultInput = $("#resultOutputPath");

  if (!samplesInput.value || samplesInput.value === previousPaths.samples) {
    samplesInput.value = nextPaths.samples;
  }
  if (!resultInput.value || resultInput.value === previousPaths.result) {
    resultInput.value = nextPaths.result;
  }
  lastRagasMode = mode;
  lastRagasStrategy = strategy;
}

function getRagasPayload() {
  const retrievalMode = $("#retrievalMode").value;
  const retrievalStrategy = $("#ragasRetrievalStrategy").value;
  const defaultPaths = getRagasDefaultPaths(retrievalMode, retrievalStrategy);
  return {
    dataset_path: $("#datasetPath").value.trim() || "eval_data/ai_agents_in_depth_eval_dataset.json",
    retrieval_mode: retrievalMode,
    retrieval_strategy: retrievalStrategy,
    samples_output_path: $("#samplesOutputPath").value.trim() || defaultPaths.samples,
    result_output_path: $("#resultOutputPath").value.trim() || defaultPaths.result,
    top_k: parseIntInput("#ragasTopK", 3),
    max_samples: parseOptionalInt("#maxSamples"),
    force_rebuild: $("#forceRebuild").checked,
  };
}

function renderRagasResult(result) {
  $("#ragasResult").textContent = JSON.stringify(
    {
      sample_count: result.sample_count,
      retrieval_mode: result.retrieval_mode,
      retrieval_strategy: result.retrieval_strategy,
      cache_used: result.cache_used,
      cache_status: result.cache_status,
      samples_path: result.samples_path,
      result_path: result.result_path,
      csv_path: result.csv_path,
      top_k: result.top_k,
      max_samples: result.max_samples,
      retrieval_metrics: result.retrieval_metrics,
      metric_null_counts: result.metric_null_counts,
      summary_path: result.summary_path,
    },
    null,
    2
  );

  const metrics = {
    ...(result.metrics || {}),
    ...(result.retrieval_metrics || {}),
  };
  $("#metricsGrid").innerHTML = Object.entries(metrics)
    .map(
      ([name, value]) => `
        <div class="metric-card">
          <span>${escapeHtml(formatRagasMetricName(name))}</span>
          <strong>${escapeHtml(value == null ? "-" : value)}</strong>
        </div>
      `
    )
    .join("");
}

async function buildRagasSamples() {
  hideNotice();
  const button = $("#buildSamplesBtn");
  setLoading(button, true, "生成中...");
  try {
    const result = await apiFetch("/api/evaluation/ragas/build-samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getRagasPayload()),
    });
    renderRagasResult(result);
    showNotice(`样本已生成：${result.sample_count} 条，缓存状态：${result.cache_status}`);
  } catch (error) {
    showNotice(`生成样本失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

async function runRagas() {
  hideNotice();
  const ok = window.confirm("RAGAS 完整评估可能较慢并消耗 DeepSeek token，确认继续？");
  if (!ok) return;

  const button = $("#runRagasBtn");
  setLoading(button, true, "评估中...");
  try {
    const result = await apiFetch("/api/evaluation/ragas/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getRagasPayload()),
    });
    renderRagasResult(result);
    showNotice(`评估完成：${result.sample_count} 条，缓存状态：${result.cache_status}`);
  } catch (error) {
    showNotice(`评估失败：${error.message}`, "error");
  } finally {
    setLoading(button, false);
  }
}

function switchTab(tabName) {
  state.currentTab = tabName;
  $$(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  $$(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}Panel`);
  });

  if (tabName === "documents") loadDocuments();
  if (tabName === "chunks") loadChunks();
}

function bindEvents() {
  $$(".tab-button").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  $("#uploadForm").addEventListener("submit", uploadDocument);
  $("#retrievalMode").addEventListener("change", syncRagasOutputPaths);
  $("#ragasRetrievalStrategy").addEventListener("change", syncRagasOutputPaths);
  $("#refreshDocumentsBtn").addEventListener("click", () => {
    refreshStatus();
    loadDocuments();
  });
  onClick("#clearVectorBtn", clearVectorStore);
  onClick("#refreshBatchHistoryBtn", loadBatchHistory);
  onClick("#askBtn", askRag);
  onClick("#streamBtn", streamRag);
  onClick("#langGraphBtn", askLangGraphAgent);
  onClick("#dynamicBtn", askDynamicRag);
  $("#clearAnswerBtn").addEventListener("click", () => {
    $("#answerText").textContent = "还没有回答。";
    $("#sourcesBox").innerHTML = "";
  });
  onClick("#loadChunksBtn", loadChunks);
  onClick("#buildSamplesBtn", buildRagasSamples);
  onClick("#runRagasBtn", runRagas);

  document.addEventListener("click", (event) => {
    const retryButton = event.target.closest(".retry-upload");
    if (retryButton) {
      retryBatchItem(retryButton.dataset.batchId, retryButton.dataset.itemId, retryButton);
      return;
    }
    const detailButton = event.target.closest(".batch-detail");
    if (detailButton) loadBatchDetail(detailButton.dataset.batchId);
    const deleteButton = event.target.closest(".delete-document");
    if (deleteButton) deleteDocument(deleteButton.dataset.documentId);
    const reindexButton = event.target.closest(".reindex-document");
    if (reindexButton) reindexDocument(reindexButton.dataset.documentId, reindexButton);
  });
}

async function init() {
  bindEvents();
  await Promise.all([refreshStatus(), loadDocuments(), loadBatchHistory()]);
}

init();
