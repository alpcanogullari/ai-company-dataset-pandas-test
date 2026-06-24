def agent_widget_styles():
    return """
        .agent-tab {
            position: relative;
            z-index: 2;
            justify-self: end;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 132px;
            min-height: 44px;
            box-sizing: border-box;
            border: 1px solid #2457d6;
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            background: #2457d6;
            color: #ffffff;
            padding: 11px 14px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(23, 32, 42, 0.16);
        }
        .agent-panel {
            position: fixed;
            right: 18px;
            bottom: 18px;
            z-index: 60;
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
            width: min(430px, calc(100vw - 28px));
            max-height: min(720px, calc(100vh - 36px));
            opacity: 1;
            pointer-events: auto;
            transform: translateY(calc(100% - 44px));
            transition: transform 220ms ease;
            overflow: visible;
        }
        .agent-panel.open {
            transform: translateY(0);
        }
        .agent-chat-shell {
            display: grid;
            grid-template-rows: auto auto auto minmax(0, 1fr) auto;
            min-height: 0;
            height: min(640px, calc(100vh - 88px));
            background: #ffffff;
            border: 1px solid #b7c2cc;
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(23, 32, 42, 0.18);
            overflow: hidden;
        }
        .agent-head {
            display: flex;
            align-items: center;
            padding: 12px 14px;
            border-bottom: 1px solid #d8dee4;
        }
        .agent-head strong {
            font-size: 15px;
        }
        .agent-close {
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
            min-height: 34px;
            padding: 6px 9px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 700;
        }
        .agent-actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 7px;
            padding: 9px 14px;
            width: 100%;
            box-sizing: border-box;
            border-bottom: 1px solid #d8dee4;
            background: rgba(255, 255, 255, 0.92);
            position: relative;
            z-index: 3;
        }
        .agent-actions .agent-close {
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .agent-history-menu {
            display: none;
            max-height: 190px;
            overflow-y: auto;
            border: 0;
            border-bottom: 1px solid #d8dee4;
            border-radius: 0;
            background: #ffffff;
            box-shadow: none;
            padding: 10px 14px;
        }
        .agent-history-menu.open {
            display: grid;
            gap: 8px;
        }
        .agent-history-item {
            border: 1px solid #d8dee4;
            border-radius: 6px;
            padding: 8px;
            font-size: 12px;
            line-height: 1.4;
            color: #17202a;
            background: #fbfcfd;
            overflow-wrap: anywhere;
        }
        .agent-history-item b {
            display: block;
            margin-bottom: 3px;
            color: #5b6670;
            font-size: 11px;
            text-transform: uppercase;
        }
        .agent-messages {
            min-height: 0;
            overflow-y: auto;
            padding: 14px;
            display: grid;
            gap: 10px;
            align-content: start;
            background: #f6f8fa;
        }
        .agent-message {
            max-width: 92%;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            padding: 10px 11px;
            background: #ffffff;
            color: #17202a;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-size: 13px;
            line-height: 1.45;
        }
        .agent-message.user {
            justify-self: end;
            background: #2457d6;
            color: #ffffff;
            border-color: #2457d6;
        }
        .agent-message.status {
            color: #5b6670;
            font-style: italic;
        }
        .agent-form {
            --agent-input-height: 44px;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: end;
            gap: 8px;
            padding: 10px 12px 7px;
            border-top: 1px solid #d8dee4;
            background: inherit;
            margin-top: auto;
        }
        .agent-input {
            width: 100%;
            height: var(--agent-input-height);
            min-height: 44px;
            max-height: 96px;
            resize: none;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            padding: 11px 10px;
            font: inherit;
            line-height: 1.35;
            color: #17202a;
            box-sizing: border-box;
            overflow-y: auto;
        }
        .agent-send {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            align-self: end;
            min-width: 72px;
            height: var(--agent-input-height);
            min-height: 44px;
            border: 1px solid #2457d6;
            border-radius: 6px;
            background: #2457d6;
            color: #ffffff;
            padding: 0 16px;
            cursor: pointer;
            font-weight: 700;
        }
        .agent-send:disabled {
            cursor: wait;
            opacity: 0.65;
        }
        .agent-tab {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.24), rgba(255, 101, 200, 0.18)), #080d16;
            border-color: rgba(96, 220, 255, 0.46);
            color: #e8f7ff;
            box-shadow: 0 0 24px rgba(54, 216, 255, 0.24), inset 0 0 18px rgba(255, 101, 200, 0.10);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }
        .agent-tab:hover {
            transform: translateY(-1px);
            box-shadow: 0 0 30px rgba(54, 216, 255, 0.34), inset 0 0 20px rgba(255, 101, 200, 0.14);
        }
        .agent-chat-shell {
            background: rgba(8, 13, 22, 0.98);
            border-color: rgba(96, 220, 255, 0.34);
            color: #e8f7ff;
            box-shadow: 0 24px 54px rgba(0, 0, 0, 0.40), 0 0 32px rgba(54, 216, 255, 0.14);
        }
        .agent-panel.open .agent-chat-shell {
            animation: agentChatIn 180ms ease both;
        }
        .agent-head, .agent-actions, .agent-form {
            border-color: rgba(96, 220, 255, 0.20);
        }
        .agent-close, .agent-send {
            background: rgba(12, 20, 33, 0.96);
            border-color: rgba(96, 220, 255, 0.52);
            color: #e8f7ff;
            box-shadow: inset 0 0 12px rgba(54, 216, 255, 0.08);
        }
        .agent-send {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.26), rgba(255, 101, 200, 0.16));
        }
        .agent-history-menu {
            background: rgba(8, 13, 22, 0.98);
            border-color: rgba(96, 220, 255, 0.20);
            box-shadow: inset 0 -10px 18px rgba(54, 216, 255, 0.04);
        }
        .agent-actions {
            background: rgba(8, 13, 22, 0.98);
        }
        .agent-history-item, .agent-message {
            background: rgba(16, 25, 40, 0.96);
            border-color: rgba(96, 220, 255, 0.22);
            color: #e8f7ff;
        }
        .agent-history-item b, .agent-message.status {
            color: #91a9bd;
        }
        .agent-messages {
            background:
                repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 80px),
                #060a12;
        }
        .agent-message.user {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.28), rgba(255, 101, 200, 0.18));
            border-color: rgba(96, 220, 255, 0.44);
            color: #ffffff;
            box-shadow: 0 0 16px rgba(54, 216, 255, 0.14);
        }
        .agent-input {
            background: #060a12;
            border-color: rgba(96, 220, 255, 0.30);
            color: #e8f7ff;
        }
        .agent-input::placeholder {
            color: #72889b;
        }
        @keyframes agentChatIn {
            from {
                opacity: 0.85;
            }
            to {
                opacity: 1;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .agent-tab, .agent-panel, .agent-panel.open .agent-chat-shell {
                animation: none;
                transition: none;
            }
        }
        @media (max-width: 520px) {
            .agent-panel {
                right: 10px;
                bottom: 10px;
                width: calc(100vw - 20px);
                max-height: calc(100vh - 20px);
            }
            .agent-chat-shell {
                height: calc(100vh - 74px);
            }
        }
    """


def agent_widget_markup():
    return """
    <aside class="agent-panel" id="agent-panel" aria-label="AI dataset agent">
        <button class="agent-tab" id="agent-tab" type="button" aria-controls="agent-chat-shell" aria-expanded="false">AI Agent</button>
        <div class="agent-chat-shell" id="agent-chat-shell">
            <div class="agent-head">
                <strong>AI Dataset Agent</strong>
            </div>
            <div class="agent-actions" aria-label="AI agent controls">
                <button class="agent-close" id="agent-history-toggle" type="button" aria-controls="agent-history-menu" aria-expanded="false">History</button>
                <button class="agent-close" id="agent-clear" type="button" title="Clear chat history">Clear</button>
                <button class="agent-close" id="agent-close" type="button">Close</button>
            </div>
            <div class="agent-history-menu" id="agent-history-menu" aria-label="Saved chat history"></div>
            <div class="agent-messages" id="agent-messages">
                <div class="agent-message">Ready. Ask about the dataset, regression, XGBoost, countries, or industries.</div>
            </div>
            <form class="agent-form" id="agent-form">
                <textarea class="agent-input" id="agent-input" rows="1" placeholder="Ask about the data"></textarea>
                <button class="agent-send" id="agent-send" type="submit">Send</button>
            </form>
        </div>
    </aside>
    """


def agent_widget_script():
    return """
        function setupAgentWidget() {
            const tab = document.getElementById("agent-tab");
            const panel = document.getElementById("agent-panel");
            const closeButton = document.getElementById("agent-close");
            const clearButton = document.getElementById("agent-clear");
            const historyToggle = document.getElementById("agent-history-toggle");
            const historyMenu = document.getElementById("agent-history-menu");
            const form = document.getElementById("agent-form");
            const input = document.getElementById("agent-input");
            const sendButton = document.getElementById("agent-send");
            const messages = document.getElementById("agent-messages");

            if (!tab || !panel || !form || !input || !messages) {
                return;
            }

            const storageKey = "ai_coverage_agent_session_id";
            let sessionId = localStorage.getItem(storageKey);
            if (!sessionId) {
                sessionId = `dashboard-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
                localStorage.setItem(storageKey, sessionId);
            }
            const messagesKey = `ai_coverage_agent_messages_${sessionId}`;

            function getLocalMessages() {
                try {
                    const parsed = JSON.parse(localStorage.getItem(messagesKey) || "[]");
                    return Array.isArray(parsed) ? parsed : [];
                } catch (error) {
                    return [];
                }
            }

            function saveLocalMessages(items) {
                localStorage.setItem(messagesKey, JSON.stringify(items.slice(-80)));
            }

            function pushLocalMessage(role, text) {
                if (!text || role === "status") {
                    return;
                }
                const savedRole = role === "user" ? "user" : "assistant";
                const items = getLocalMessages();
                items.push({ role: savedRole, content: text });
                saveLocalMessages(items);
                renderHistoryMenu();
            }

            function renderMessages(items) {
                messages.innerHTML = "";
                if (!items.length) {
                    appendMessage("", "Ready. Ask about the dataset, regression, XGBoost, countries, or industries.", false);
                    return;
                }
                items.forEach((item) => {
                    const role = item.role === "user" ? "user" : "";
                    appendMessage(role, item.content || "", false);
                });
            }

            function renderHistoryMenu() {
                if (!historyMenu) {
                    return;
                }
                const items = getLocalMessages();
                if (!items.length) {
                    historyMenu.innerHTML = '<div class="agent-history-item">No saved messages yet.</div>';
                    return;
                }
                historyMenu.innerHTML = items.map((item) => {
                    const role = item.role === "user" ? "You" : "Agent";
                    const text = String(item.content || "")
                        .replace(/[*]{2}(.+?)[*]{2}|__(.+?)__/g, "$1$2")
                        .slice(0, 360);
                    const safeRole = role.replace(/[&<>"']/g, "");
                    const safeText = text
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#39;");
                    return `<div class="agent-history-item"><b>${safeRole}</b>${safeText}</div>`;
                }).join("");
            }

            function loadLocalHistory() {
                const items = getLocalMessages();
                if (items.length) {
                    renderMessages(items);
                }
                renderHistoryMenu();
            }

            async function loadStoredHistory() {
                loadLocalHistory();
                try {
                    const response = await fetch("/api/agent", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id: sessionId, history: true })
                    });
                    if (!response.ok) {
                        return;
                    }
                    const payload = await response.json();
                    const storedMessages = Array.isArray(payload.messages) ? payload.messages : [];
                    if (!storedMessages.length) {
                        return;
                    }
                    saveLocalMessages(storedMessages);
                    renderMessages(storedMessages);
                    renderHistoryMenu();
                } catch (error) {
                    return;
                }
            }

            function setOpen(isOpen) {
                panel.classList.toggle("open", isOpen);
                tab.classList.toggle("open", isOpen);
                tab.setAttribute("aria-expanded", String(isOpen));
                if (isOpen) {
                    input.focus();
                }
            }

            function setHistoryOpen(isOpen) {
                if (!historyMenu || !historyToggle) {
                    return;
                }
                historyMenu.classList.toggle("open", isOpen);
                panel.classList.toggle("history-open", isOpen);
                historyToggle.setAttribute("aria-expanded", String(isOpen));
            }

            function appendFormattedText(target, text, allowFormatting) {
                if (!allowFormatting) {
                    target.textContent = text;
                    return;
                }

                const pattern = /[*]{2}(.+?)[*]{2}|__(.+?)__/g;
                let cursor = 0;
                let match;
                while ((match = pattern.exec(text)) !== null) {
                    if (match.index > cursor) {
                        target.appendChild(document.createTextNode(text.slice(cursor, match.index)));
                    }
                    const strong = document.createElement("strong");
                    strong.textContent = match[1] || match[2];
                    target.appendChild(strong);
                    cursor = pattern.lastIndex;
                }
                if (cursor < text.length) {
                    target.appendChild(document.createTextNode(text.slice(cursor)));
                }
            }

            function appendMessage(role, text, save = false) {
                const message = document.createElement("div");
                message.className = `agent-message ${role}`;
                appendFormattedText(message, text, role !== "user" && role !== "status");
                messages.appendChild(message);
                messages.scrollTop = messages.scrollHeight;
                if (save) {
                    pushLocalMessage(role, text);
                }
                return message;
            }

            function startProgress(message) {
                const steps = [
                    "Reading local dataset context...",
                    "Checking regression outputs...",
                    "Reviewing chat history...",
                    "Preparing a concise answer..."
                ];
                let index = 0;
                message.textContent = steps[index];
                return window.setInterval(() => {
                    index = (index + 1) % steps.length;
                    message.textContent = steps[index];
                }, 1200);
            }

            function stopProgress(message, timer) {
                if (timer) {
                    window.clearInterval(timer);
                }
                if (message && message.isConnected) {
                    message.remove();
                }
            }

            function resizeInput() {
                input.style.height = "44px";
                const height = Math.min(Math.max(input.scrollHeight, 44), 96);
                input.style.height = `${height}px`;
                form.style.setProperty("--agent-input-height", `${height}px`);
            }

            tab.addEventListener("click", () => {
                const isOpen = !panel.classList.contains("open");
                if (!isOpen) {
                    setHistoryOpen(false);
                }
                setOpen(isOpen);
            });
            closeButton.addEventListener("click", () => {
                setHistoryOpen(false);
                setOpen(false);
            });
            historyToggle.addEventListener("click", () => {
                renderHistoryMenu();
                setHistoryOpen(!historyMenu.classList.contains("open"));
            });
            clearButton.addEventListener("click", async () => {
                const previousLabel = clearButton.textContent;
                clearButton.disabled = true;
                clearButton.textContent = "Clearing...";
                input.value = "";
                resizeInput();
                setHistoryOpen(false);
                messages.innerHTML = "";
                saveLocalMessages([]);
                renderHistoryMenu();
                appendMessage("", "Ready. Ask about the dataset, regression, XGBoost, countries, or industries.", false);
                try {
                    await fetch("/api/agent", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id: sessionId, reset: true })
                    });
                } catch (error) {
                    appendMessage("", "Local visible chat was cleared. Start the dashboard server to clear stored history.");
                } finally {
                    clearButton.disabled = false;
                    clearButton.textContent = previousLabel;
                }
            });
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape") {
                    setHistoryOpen(false);
                    setOpen(false);
                }
            });
            input.addEventListener("keydown", (event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    form.requestSubmit();
                }
            });
            input.addEventListener("input", resizeInput);

            form.addEventListener("submit", async (event) => {
                event.preventDefault();
                const question = input.value.trim();
                if (!question) {
                    return;
                }

                setHistoryOpen(false);
                appendMessage("user", question, true);
                input.value = "";
                resizeInput();
                sendButton.disabled = true;
                const status = appendMessage("status", "Reading local dataset context...");
                const progressTimer = startProgress(status);

                try {
                    const response = await fetch("/api/agent", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id: sessionId, question })
                    });
                    const payload = await response.json();
                    stopProgress(status, progressTimer);

                    if (!response.ok) {
                        appendMessage("", payload.error || "The agent request failed.");
                    } else {
                        appendMessage("", payload.answer || "No answer returned.", true);
                    }
                } catch (error) {
                    stopProgress(status, progressTimer);
                    appendMessage("", "Start the dashboard server to use the AI agent.");
                } finally {
                    sendButton.disabled = false;
                    input.focus();
                }
            });

            loadStoredHistory();
            resizeInput();
        }
    """
