const apiBase = document.querySelector("#apiBase");
const sessionId = document.querySelector("#sessionId");
const token = document.querySelector("#token");
const loadBtn = document.querySelector("#loadBtn");
const clearBtn = document.querySelector("#clearBtn");
const statusEl = document.querySelector("#status");
const chatHistory = document.querySelector("#chatHistory");
const messageCount = document.querySelector("#messageCount");

function setStatus(text) {
  statusEl.textContent = text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function normalizeRole(role) {
  if (role === "bot") return "assistant";
  return role || "message";
}

function renderSources(citations) {
  const sources = citations?.sources || [];
  if (!sources.length) return "";

  const items = sources.slice(0, 4).map((source) => {
    const label = source.type === "web"
      ? source.title || source.url || "Web source"
      : `Page ${source.page ?? source.page_label ?? "n/a"}`;
    return `<li>${escapeHtml(label)}</li>`;
  }).join("");

  return `
    <details class="bubble-details">
      <summary>Sources (${sources.length})</summary>
      <ul>${items}</ul>
    </details>
  `;
}

function renderMessage(message) {
  const role = normalizeRole(message.role);
  const isUser = role === "user";
  const confidence = typeof message.confidence === "number"
    ? `${Math.round(message.confidence * 100)}%`
    : "n/a";

  const meta = isUser ? "" : `
    <div class="bubble-meta">
      <span>confidence: ${escapeHtml(confidence)}</span>
      ${message.tool_used ? `<span>tool: ${escapeHtml(message.tool_used)}</span>` : ""}
      ${message.used_vector_db ? "<span>vector: yes</span>" : ""}
    </div>
    ${renderSources(message.citations)}
  `;

  return `
    <article class="chat-row ${isUser ? "user-row" : "assistant-row"}">
      <div class="bubble ${isUser ? "user-bubble" : "assistant-bubble"}">
        <div class="bubble-role">${escapeHtml(isUser ? "You" : "DocuMind")}</div>
        <div class="bubble-content">${escapeHtml(message.content)}</div>
        ${meta}
        <div class="bubble-time">${escapeHtml(formatDate(message.created_at))}</div>
      </div>
    </article>
  `;
}

function renderHistory(messages) {
  messageCount.textContent = `${messages.length} message${messages.length === 1 ? "" : "s"}`;
  chatHistory.className = messages.length ? "chat-history" : "chat-history empty";
  chatHistory.innerHTML = messages.length
    ? messages.map(renderMessage).join("")
    : "No messages found for this session.";
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function loadDevConfig() {
  try {
    const response = await fetch("/frontend/dev-config.json", {
      cache: "no-store",
    });
    if (!response.ok) return;

    const config = await response.json();
    if (config.apiBase) apiBase.value = config.apiBase;
    if (config.sessionId) sessionId.value = config.sessionId;
    if (config.bearerToken) token.value = config.bearerToken;
    setStatus("Loaded local dev config.");
  } catch {
    // Optional local config.
  }
}

async function loadHistory() {
  const base = apiBase.value.trim().replace(/\/$/, "");
  const sid = sessionId.value.trim();
  const authToken = token.value.trim();

  if (!base || !sid || !authToken) {
    setStatus("Fill API base URL, session ID, and bearer token.");
    return;
  }

  loadBtn.disabled = true;
  setStatus("Loading history...");

  try {
    const response = await fetch(`${base}/sessions/${encodeURIComponent(sid)}/history`, {
      headers: {
        "Authorization": `Bearer ${authToken}`,
        "Accept": "application/json",
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${errorText}`);
    }

    const data = await response.json();
    renderHistory(data.messages || []);
    setStatus("Loaded.");
  } catch (error) {
    setStatus(error.message);
  } finally {
    loadBtn.disabled = false;
  }
}

loadBtn.addEventListener("click", loadHistory);
clearBtn.addEventListener("click", () => {
  chatHistory.className = "chat-history empty";
  chatHistory.textContent = "Load a session to view messages.";
  messageCount.textContent = "0 messages";
  setStatus("Idle");
});

loadDevConfig();
