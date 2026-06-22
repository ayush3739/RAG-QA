const apiBase = document.querySelector("#apiBase");
const sessionId = document.querySelector("#sessionId");
const token = document.querySelector("#token");
const question = document.querySelector("#question");
const sendBtn = document.querySelector("#sendBtn");
const clearBtn = document.querySelector("#clearBtn");
const statusEl = document.querySelector("#status");
const answerEl = document.querySelector("#answer");
const sourcesEl = document.querySelector("#sources");
const chunksEl = document.querySelector("#chunks");
const toolTraceEl = document.querySelector("#toolTrace");
const confidenceEl = document.querySelector("#confidence");
const elapsedTimeEl = document.querySelector("#elapsedTime");
let activeTimer = null;
let requestStartedAt = 0;

async function loadDevConfig() {
  try {
    const response = await fetch("/frontend/dev-config.json", {
      cache: "no-store",
    });

    if (!response.ok) {
      return;
    }

    const config = await response.json();
    if (config.apiBase) apiBase.value = config.apiBase;
    if (config.sessionId) sessionId.value = config.sessionId;
    if (config.bearerToken) token.value = config.bearerToken;
    if (config.question) question.value = config.question;
    setStatus("Loaded local dev config.");
  } catch {
    // Missing or invalid dev config should not block manual testing.
  }
}

function setStatus(text) {
  statusEl.textContent = text;
}

function resetOutput() {
  if (activeTimer) {
    clearInterval(activeTimer);
    activeTimer = null;
  }
  answerEl.textContent = "";
  sourcesEl.className = "cards empty";
  sourcesEl.textContent = "Sources appear after metadata arrives.";
  chunksEl.className = "cards empty";
  chunksEl.textContent = "Chunks appear after metadata arrives.";
  toolTraceEl.textContent = "No tool yet";
  confidenceEl.textContent = "confidence: n/a";
  elapsedTimeEl.textContent = "time: n/a";
}

function formatElapsed(ms) {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

function startTimer() {
  requestStartedAt = performance.now();
  elapsedTimeEl.textContent = "time: 0 ms";
  activeTimer = setInterval(() => {
    elapsedTimeEl.textContent = `time: ${formatElapsed(performance.now() - requestStartedAt)}`;
  }, 100);
}

function stopTimer() {
  if (activeTimer) {
    clearInterval(activeTimer);
    activeTimer = null;
  }
  if (requestStartedAt) {
    elapsedTimeEl.textContent = `time: ${formatElapsed(performance.now() - requestStartedAt)}`;
  }
}

function renderSources(sources = []) {
  sourcesEl.className = sources.length ? "cards" : "cards empty";
  sourcesEl.innerHTML = "";

  if (!sources.length) {
    sourcesEl.textContent = "No sources returned.";
    return;
  }

  for (const source of sources) {
    const card = document.createElement("article");
    card.className = "card";
    const title = source.type === "web"
      ? source.title || source.url || "Web source"
      : `Document source${source.page ? ` - Page ${source.page}` : ""}`;
    card.innerHTML = `
      <h3>${escapeHtml(title)}</h3>
      <p class="mono">${escapeHtml(source.url || source.source || source.chunk_id || "")}</p>
      <p>${escapeHtml(source.excerpt || "")}</p>
    `;
    sourcesEl.appendChild(card);
  }
}

function renderChunks(chunks = []) {
  chunksEl.className = chunks.length ? "cards" : "cards empty";
  chunksEl.innerHTML = "";

  if (!chunks.length) {
    chunksEl.textContent = "No retrieved document chunks returned.";
    return;
  }

  for (const chunk of chunks) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <h3>Page ${escapeHtml(String(chunk.page ?? "n/a"))}</h3>
      <p class="mono">${escapeHtml(chunk.source || chunk.chunk_id || "")}</p>
      <p>${escapeHtml(chunk.text || "")}</p>
    `;
    chunksEl.appendChild(card);
  }
}

function renderMetadata(metadata) {
  toolTraceEl.textContent = (metadata.tool_trace || []).join(" -> ") || "No tool";
  const confidence = metadata.confidence;
  confidenceEl.textContent = typeof confidence === "number"
    ? `confidence: ${Math.round(confidence * 100)}%`
    : "confidence: n/a";
  renderSources(metadata.sources || []);
  renderChunks(metadata.chunks || []);

  if (metadata.routing_reason) {
    setStatus(`Done. Routing: ${metadata.routing_reason}`);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseSseEvents(buffer) {
  const events = [];
  const parts = buffer.split(/\r?\n\r?\n/);
  const remainder = parts.pop() || "";

  for (const part of parts) {
    let event = "message";
    const dataLines = [];

    for (const line of part.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }

    events.push({ event, data: dataLines.join("\n") });
  }

  return { events, remainder };
}

async function sendQuestion() {
  resetOutput();

  const base = apiBase.value.trim().replace(/\/$/, "");
  const sid = sessionId.value.trim();
  const authToken = token.value.trim();
  const text = question.value.trim();

  if (!base || !sid || !authToken || !text) {
    setStatus("Fill API base URL, session ID, bearer token, and question.");
    return;
  }

  sendBtn.disabled = true;
  setStatus("Connecting...");
  startTimer();

  try {
    const response = await fetch(`${base}/chat/${encodeURIComponent(sid)}`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${authToken}`,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify({ question: text }),
    });

    if (!response.ok || !response.body) {
      const errorText = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${errorText}`);
    }

    setStatus("Streaming answer...");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseEvents(buffer);
      buffer = parsed.remainder;

      for (const sse of parsed.events) {
        if (sse.event === "token") {
          answerEl.textContent += sse.data;
        } else if (sse.event === "metadata") {
          renderMetadata(JSON.parse(sse.data));
        } else if (sse.event === "error") {
          stopTimer();
          setStatus(`Error: ${sse.data}`);
        } else if (sse.event === "done") {
          stopTimer();
          setStatus("Done.");
        }
      }
    }
  } catch (error) {
    setStatus(error.message);
  } finally {
    stopTimer();
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendQuestion);
clearBtn.addEventListener("click", () => {
  resetOutput();
  setStatus("Idle");
});

resetOutput();
loadDevConfig();
