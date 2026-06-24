import { readFile } from "node:fs/promises";
import { join } from "node:path";

const CONTEXT_FILES = [
  "aicoveragedata/site/agent_context.md",
  "README.md",
];

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

function json(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: JSON_HEADERS,
  });
}

async function readContext() {
  const chunks = [];
  for (const file of CONTEXT_FILES) {
    try {
      const text = await readFile(join(process.cwd(), file), "utf8");
      chunks.push(`--- ${file} ---\n${text.slice(0, 16000)}`);
    } catch {
      // The function can still answer from the remaining context.
    }
  }
  return chunks.join("\n\n");
}

function cleanMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages
    .filter((item) => item && typeof item.content === "string")
    .slice(-12)
    .map((item) => {
      const role = item.role === "user" ? "User" : "Agent";
      return `${role}: ${item.content.slice(0, 2000)}`;
    });
}

function extractText(payload) {
  if (payload.output_text) {
    return payload.output_text;
  }

  const parts = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (content.text) {
        parts.push(content.text);
      }
    }
  }
  return parts.join("\n").trim();
}

export default async (request) => {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204 });
  }

  if (request.method !== "POST") {
    return json(405, { error: "Use POST for /api/agent." });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json(400, { error: "Request body must be valid JSON." });
  }

  const sessionId = String(body.session_id || "dashboard").trim() || "dashboard";

  if (body.reset) {
    return json(200, { answer: "Chat history cleared.", session_id: sessionId });
  }

  if (body.history) {
    return json(200, { messages: [], session_id: sessionId });
  }

  const question = String(body.question || "").trim();
  if (!question) {
    return json(400, { error: "Question is required." });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return json(500, {
      error: "OPENAI_API_KEY is missing in Netlify environment variables.",
    });
  }

  const context = await readContext();
  const history = cleanMessages(body.client_messages).join("\n");
  const model = process.env.OPENAI_MODEL || "gpt-5.2";

  const input = [
    history ? `Conversation history:\n${history}` : "Conversation history: none",
    `User question:\n${question}`,
    `Verified dashboard context:\n${context}`,
  ].join("\n\n");

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      instructions:
        "You are the AI Coverage Dashboard agent. Answer using the provided dashboard context. Be short and concrete. Include examples when useful. If context is insufficient, say so plainly.",
      input,
      max_output_tokens: 800,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    return json(response.status, {
      error: payload.error?.message || "OpenAI request failed.",
      session_id: sessionId,
    });
  }

  return json(200, {
    answer: extractText(payload) || "No answer returned.",
    session_id: sessionId,
  });
};
