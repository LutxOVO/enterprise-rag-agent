const state = {
  currentTab: "documents",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

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
  tbody.innerHTML = `<tr><td colspan="5">加载中...</td></tr>`;
  try {
    const docs = await apiFetch("/api/documents");
    if (!docs.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-state">暂无文档。</td></tr>`;
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
          </tr>
        `
      )
      .join("");
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-state">加载失败：${escapeHtml(error.message)}</td></tr>`;
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  hideNotice();
  const file = $("#uploadFile").files[0];
  if (!file) {
    showNotice("请先选择文件。", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  const button = $("#uploadBtn");
  setLoading(button, true, "上传中...");

  try {
    const result = await apiFetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    $("#uploadResult").textContent = JSON.stringify(result, null, 2);
    showNotice(`上传成功：${result.filename}，生成 ${result.chunk_count} 个 chunk。`);
    await Promise.all([refreshStatus(), loadDocuments()]);
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
    await Promise.all([refreshStatus(), loadDocuments(), loadChunks()]);
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
            <span class="source-meta">切片 ${escapeHtml(source.chunk_index)} / 得分 ${escapeHtml(source.score)}</span>
          </summary>
          <pre>${escapeHtml(source.content_preview)}</pre>
        </details>
      `
    )
    .join("");
}

function renderDynamicExtra(extra = {}) {
  const chunks = extra.retrieved_chunks || [];
  const queryInfo = `
Dynamic RAG 调试信息

原始问题 (original_query):
${extra.original_query || ""}

问题重写 (rewritten_query):
${extra.rewritten_query || ""}

HyDE 虚构答案 (hyde_answer):
${extra.hyde_answer || ""}

实际检索文本 (retrieval_query):
${extra.retrieval_query || ""}
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

function getRagasPayload() {
  return {
    dataset_path: $("#datasetPath").value.trim() || "eval_data/ragas_eval_dataset.json",
    samples_output_path: $("#samplesOutputPath").value.trim() || "eval_outputs/ragas_samples.json",
    result_output_path: $("#resultOutputPath").value.trim() || "eval_outputs/ragas_result.json",
    top_k: parseIntInput("#ragasTopK", 3),
    max_samples: parseOptionalInt("#maxSamples"),
    force_rebuild: $("#forceRebuild").checked,
  };
}

function renderRagasResult(result) {
  $("#ragasResult").textContent = JSON.stringify(
    {
      sample_count: result.sample_count,
      cache_used: result.cache_used,
      cache_status: result.cache_status,
      samples_path: result.samples_path,
      result_path: result.result_path,
      csv_path: result.csv_path,
      top_k: result.top_k,
      max_samples: result.max_samples,
    },
    null,
    2
  );

  const metrics = result.metrics || {};
  $("#metricsGrid").innerHTML = Object.entries(metrics)
    .map(
      ([name, value]) => `
        <div class="metric-card">
          <span>${escapeHtml(formatRagasMetricName(name))}</span>
          <strong>${escapeHtml(value)}</strong>
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
  $("#refreshDocumentsBtn").addEventListener("click", () => {
    refreshStatus();
    loadDocuments();
  });
  $("#clearVectorBtn").addEventListener("click", clearVectorStore);
  $("#askBtn").addEventListener("click", askRag);
  $("#streamBtn").addEventListener("click", streamRag);
  $("#dynamicBtn").addEventListener("click", askDynamicRag);
  $("#clearAnswerBtn").addEventListener("click", () => {
    $("#answerText").textContent = "还没有回答。";
    $("#sourcesBox").innerHTML = "";
  });
  $("#loadChunksBtn").addEventListener("click", loadChunks);
  $("#buildSamplesBtn").addEventListener("click", buildRagasSamples);
  $("#runRagasBtn").addEventListener("click", runRagas);
}

async function init() {
  bindEvents();
  await Promise.all([refreshStatus(), loadDocuments()]);
}

init();
