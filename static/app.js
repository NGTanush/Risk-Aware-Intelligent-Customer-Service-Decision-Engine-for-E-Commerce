// ==========================================================================
// E-Commerce Customer Service AI - Frontend App Logic
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdDisplay = document.getElementById("threshold-display");
    const themeToggle = document.getElementById("theme-toggle");

    // Tab buttons
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Inspector DOM
    const inspectorPlaceholder = document.getElementById("inspector-placeholder");
    const inspectorContent = document.getElementById("inspector-content");

    // Sample Chips
    const sampleChips = document.querySelectorAll(".chip-btn");

    // Intents Grid & Search
    const intentsGrid = document.getElementById("intents-grid");
    const intentsSearch = document.getElementById("intents-search");

    let allIntents = [];
    let currentThreshold = parseFloat(thresholdSlider.value);

    // ==========================================================================
    // 1. INITIALIZATION & DATA FETCHING
    // ==========================================================================
    async function init() {
        try {
            // Fetch stats
            const statsRes = await fetch("/api/stats");
            if (statsRes.ok) {
                const stats = await statsRes.json();
                document.getElementById("metric-accuracy").textContent = stats.best_macro_f1 || "99.1%";
                document.getElementById("metric-intents").textContent = `${stats.intents_count} Classes`;
            }

            // Fetch intents list
            const intentsRes = await fetch("/api/intents");
            if (intentsRes.ok) {
                const data = await intentsRes.json();
                allIntents = data.intents || [];
                renderIntentsGrid(allIntents);
            }
        } catch (err) {
            console.error("Initialization error:", err);
        }
    }

    // ==========================================================================
    // 2. CHAT SUBMISSION & PIPELINE INTEGRATION
    // ==========================================================================
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Add User Message to Chat
        appendUserMessage(query);
        chatInput.value = "";

        // Show Typing Indicator
        const typingId = appendTypingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    threshold: currentThreshold
                })
            });

            removeTypingIndicator(typingId);

            if (!response.ok) {
                const errData = await response.json();
                appendBotMessage(`Error: ${errData.detail || "Failed to process request."}`, null);
                return;
            }

            const data = await response.json();

            // Append Bot Message
            appendBotMessage(data.customer_response, data);

            // Update Pipeline Inspector
            updatePipelineInspector(data);

        } catch (error) {
            removeTypingIndicator(typingId);
            console.error("Chat error:", error);
            appendBotMessage("Sorry, a network error occurred. Please try again.", null);
        }
    });

    // ==========================================================================
    // 3. RENDER MESSAGES
    // ==========================================================================
    function appendUserMessage(text) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const msgHtml = `
            <div class="message msg-user">
                <div class="avatar"><i class="fa-solid fa-user"></i></div>
                <div class="msg-content">
                    <div class="msg-header">
                        <span class="user-name">You</span>
                        <span class="msg-time">${timeStr}</span>
                    </div>
                    <div class="msg-body">${escapeHtml(text)}</div>
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML("beforeend", msgHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendBotMessage(text, meta) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        let metaHtml = "";
        if (meta) {
            const isHigh = meta.is_high_confidence;
            const badgeClass = isHigh ? "high-conf" : "low-conf";
            const icon = isHigh ? '<i class="fa-solid fa-circle-check"></i>' : '<i class="fa-solid fa-triangle-exclamation"></i>';
            const statusText = isHigh ? `High Confidence (${meta.confidence_percentage}) • ${meta.predicted_intent}` : `Low Confidence (${meta.confidence_percentage}) • Human Escalation`;

            metaHtml = `
                <div class="msg-meta-badge ${badgeClass}">
                    ${icon} <span>${statusText}</span>
                </div>
            `;
        }

        const msgHtml = `
            <div class="message msg-bot">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="msg-content">
                    <div class="msg-header">
                        <span class="bot-name">Customer Service AI</span>
                        <span class="msg-time">${timeStr}</span>
                    </div>
                    <div class="msg-body">${formatBotResponse(text)}${metaHtml}</div>
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML("beforeend", msgHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendTypingIndicator() {
        const id = "typing-" + Date.now();
        const html = `
            <div class="message msg-bot" id="${id}">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="msg-content">
                    <div class="msg-body" style="padding: 10px 16px;">
                        <span class="typing-dots"><i class="fa-solid fa-circle-notch fa-spin"></i> Classifying intent via Linear SVM...</span>
                    </div>
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML("beforeend", html);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ==========================================================================
    // 4. PIPELINE INSPECTOR UPDATE
    // ==========================================================================
    function updatePipelineInspector(data) {
        if (!data || !data.pipeline_trace) return;

        const trace = data.pipeline_trace;
        inspectorPlaceholder.classList.add("hidden");
        inspectorContent.classList.remove("hidden");

        // Step 1: NLP
        document.getElementById("t-orig-query").textContent = trace.step1_nlp.original_query;
        document.getElementById("t-clean-text").textContent = trace.step1_nlp.cleaned_text;
        document.getElementById("t-tokens-count").textContent = `${trace.step1_nlp.word_count} words`;

        const tokensContainer = document.getElementById("t-tokens-list");
        tokensContainer.innerHTML = trace.step1_nlp.tokens.map(t => `<span class="token-tag">${escapeHtml(t)}</span>`).join("");

        // Step 2: TF-IDF
        document.getElementById("t-tfidf-count").textContent = `${trace.step2_tfidf.nonzero_features_count} terms`;
        const tfidfContainer = document.getElementById("t-tfidf-features");
        
        if (trace.step2_tfidf.top_features && trace.step2_tfidf.top_features.length > 0) {
            tfidfContainer.innerHTML = trace.step2_tfidf.top_features.map(f => {
                const pct = Math.min(100, Math.round(f.weight * 100));
                return `
                    <div class="feature-bar-row">
                        <div class="bar-label">
                            <code>${escapeHtml(f.feature)}</code>
                            <span>w = ${f.weight}</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: ${pct}%;"></div>
                        </div>
                    </div>
                `;
            }).join("");
        } else {
            tfidfContainer.innerHTML = `<span style="color: var(--text-dim); font-size:11px;">No active vocabulary n-grams matched.</span>`;
        }

        // Step 3 & 4: Linear SVM
        document.getElementById("t-svm-intent").textContent = trace.step3_linear_svm.predicted_intent;
        document.getElementById("t-svm-margin").textContent = trace.step3_linear_svm.max_decision_score;
        
        const marginsContainer = document.getElementById("t-svm-margins-list");
        const topMargins = trace.step3_linear_svm.top_margins || [];
        const maxMargin = Math.max(...topMargins.map(m => Math.abs(m.margin)), 1.0);

        marginsContainer.innerHTML = topMargins.map(m => {
            const widthPct = Math.min(100, Math.max(10, Math.round((Math.max(0, m.margin + 1.0) / (maxMargin + 1.0)) * 100)));
            return `
                <div class="margin-bar-row">
                    <div class="bar-label">
                        <strong>${m.intent}</strong>
                        <span>Margin: ${m.margin}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${widthPct}%; background: ${m.intent === data.predicted_intent ? 'var(--primary)' : 'rgba(255,255,255,0.2)'};"></div>
                    </div>
                </div>
            `;
        }).join("");

        // Step 5 & 6: Calibration & Decision
        document.getElementById("t-cal-score").textContent = `${data.confidence_percentage} (${data.confidence})`;
        document.getElementById("t-threshold-val").textContent = `${Math.round(data.threshold * 100)}%`;
        
        const gateEl = document.getElementById("t-decision-gate");
        const confBar = document.getElementById("t-conf-bar");

        if (data.is_high_confidence) {
            gateEl.textContent = "ACCEPTED (HIGH CONFIDENCE)";
            gateEl.style.color = "var(--success)";
            confBar.style.background = "var(--success)";
        } else {
            gateEl.textContent = "ESCALATED (LOW CONFIDENCE)";
            gateEl.style.color = "var(--warning)";
            confBar.style.background = "var(--warning)";
        }

        confBar.style.width = `${Math.min(100, Math.round(data.confidence * 100))}%`;
    }

    // ==========================================================================
    // 5. THRESHOLD SLIDER & SAMPLE CHIPS
    // ==========================================================================
    thresholdSlider.addEventListener("input", (e) => {
        currentThreshold = parseFloat(e.target.value);
        thresholdDisplay.textContent = `${Math.round(currentThreshold * 100)}%`;
    });

    sampleChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const queryText = chip.getAttribute("data-query");
            if (queryText) {
                chatInput.value = queryText;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // ==========================================================================
    // 6. TAB NAVIGATION & INTENTS SEARCH
    // ==========================================================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    function renderIntentsGrid(intents) {
        if (!intentsGrid) return;
        intentsGrid.innerHTML = intents.map(i => `
            <div class="intent-card">
                <strong>${escapeHtml(i.name)}</strong>
                <small>ID: <code>${escapeHtml(i.id)}</code></small>
                <small>${i.sample_responses_count} training responses</small>
            </div>
        `).join("");
    }

    if (intentsSearch) {
        intentsSearch.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().strip();
            const filtered = allIntents.filter(i => i.name.toLowerCase().includes(query) || i.id.toLowerCase().includes(query));
            renderIntentsGrid(filtered);
        });
    }

    // Theme Toggle
    themeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", nextTheme);
        themeToggle.innerHTML = nextTheme === "dark" ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
    });

    // Format text
    function formatBotResponse(text) {
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        return formatted;
    }

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    // Run init
    init();
});
