# 🤝 HalluciGuard Team Collaboration & Contribution Guide

Welcome to the **HalluciGuard** project! This guide explains how our 5-agent architecture is structured, how data flows between agents, and the exact step-by-step Git workflow for each team member to work cleanly on their assigned agent without conflicting with others.

---

## 🏗️ 1. Architecture & Agent Ownership Overview

HalluciGuard is built as a **5-agent trust layer pipeline** for LLM hallucination detection and correction:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HalluciGuard Pipeline                          │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌─────────────┐  │
│  │   Detector   │──▶│   Verifier   │──▶│   Judge   │──▶│  Corrector  │  │
│  │    Agent     │   │    Agent     │   │   Agent   │   │    Agent    │  │
│  │ (Port 8001)  │   │ (Port 8002)  │   │(Port 8003)│   │ (Port 8004) │  │
│  └──────────────┘   └──────────────┘   └───────────┘   └─────────────┘  │
│         │                  │                 │                │         │
│         └──────────────────┴─────────────────┴────────────────┘         │
│                                    │                                    │
│                              ┌─────▼─────┐                              │
│                              │  Memory   │                              │
│                              │   Agent   │                              │
│                              │(Port 8005)│                              │
│                              └───────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 👥 Team Member Assignments & Directory Boundaries

Each agent lives in its own isolated folder under `agents/`. **You must ONLY modify files inside your assigned folder.**

| Agent | Directory | Assigned Feature Branch | Assigned Port | Status | Responsibilities |
|-------|-----------|-------------------------|---------------|--------|------------------|
| 🔍 **Detector Agent** | `agents/detector_agent/` | `detector-agent` | 8001 | 🟡 Open | Extract claims from LLM output, score suspicion, classify domain |
| ✅ **Verifier Agent** | `agents/verifier_agent/` | `verifier-agent` | 8002 | 🟢 Complete | Multi-source evidence retrieval, DeBERTa NLI scoring, explanations |
| ⚖️ **Judge Agent** | `agents/judge_agent/` | `judge-agent` | 8003 | 🟡 Open | Risk-weighted accept/reject/flag decision engine |
| ✏️ **Corrector Agent** | `agents/corrector_agent/` | `corrector-agent` | 8004 | 🟡 Open | Fact-based text rewriter using verified evidence |
| 🧠 **Memory Agent** | `agents/memory_agent/` | `memory-agent` | 8005 | 🟡 Open | Persistent knowledge graph & cross-session learning |

---

## 🚫 2. Rules of Isolation (How Not to Break Other Code)

1. **Strict Folder Boundaries**:
   - If you are building **Detector Agent**, edit files ONLY inside `agents/detector_agent/`.
   - Do **NOT** edit files in `agents/verifier_agent/` or root config files unless agreed upon by the team.
2. **Never Commit Directly to `main`**:
   - Always work on your assigned feature branch (`detector-agent`, `judge-agent`, etc.).
3. **Use Shared Contracts (`schemas/models.py`)**:
   - Communicate between agents using standard JSON payloads over HTTP.
   - The contract schemas are defined in `agents/verifier_agent/schemas/models.py`.
4. **Independent Virtual Environments**:
   - Keep your agent's dependencies inside your agent folder's `requirements.txt`.

---

## 🔄 3. Step-by-Step Developer Workflow

### Step 1: Clone the Repository
```bash
git clone https://github.com/Manju10092006/HalluciGuard.git
cd HalluciGuard
```

### Step 2: Switch to Your Agent's Feature Branch
Depending on your assigned agent, checkout your dedicated branch:

```bash
# If working on Detector Agent:
git checkout -b detector-agent

# If working on Judge Agent:
git checkout -b judge-agent

# If working on Corrector Agent:
git checkout -b corrector-agent

# If working on Memory Agent:
git checkout -b memory-agent
```

### Step 3: Set Up Your Agent Environment
Navigate to your agent's directory and create your code structure:

```bash
cd agents/<your_agent_directory>
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 4: Develop Your Agent
Read the `README.md` inside your agent's folder for exact specifications:
- `agents/detector_agent/README.md`
- `agents/judge_agent/README.md`
- `agents/corrector_agent/README.md`
- `agents/memory_agent/README.md`

### Step 5: Test Your Agent Locally
Ensure your agent runs independently on its designated port:
```bash
uvicorn api.main:app --port <YOUR_ASSIGNED_PORT> --reload
```

### Step 6: Commit and Push Your Changes
Only stage files inside your agent's directory:

```bash
# Example for Detector Agent:
git add agents/detector_agent/
git commit -m "feat(detector): implement perplexity claim extractor and domain router"

# Push to your feature branch on GitHub
git push -u origin <your_branch_name>
```

### Step 7: Create a Pull Request (PR)
1. Go to [github.com/Manju10092006/HalluciGuard](https://github.com/Manju10092006/HalluciGuard)
2. Click **Pull Requests** → **New Pull Request**
3. Select `base: main` ← `compare: <your-branch-name>`
4. Title your PR clearly (e.g. `feat(detector): implement claim detection engine`)
5. Request team review before merging into `main`.

---

## 🔌 4. Inter-Agent Data Flow Contracts

Agents communicate sequentially via HTTP POST endpoints. Below is the contract flow:

```
[LLM Raw Output]
       │
       ▼  (POST http://localhost:8001/detect)
┌────────────────┐
│ Detector Agent │  Returns list of SuspiciousClaim: { "claim_id": "c1", "text": "...", "domain": "healthcare" }
└────────────────┘
       │
       ▼  (POST http://localhost:8002/verify)
┌────────────────┐
│ Verifier Agent │  Returns VerifierOutputV2: { "claim_evidence": [...], "trust_score": 0.85 }
└────────────────┘
       │
       ▼  (POST http://localhost:8003/judge)
┌────────────────┐
│  Judge Agent   │  Returns Verdict: ACCEPT / REJECT / FLAG
└────────────────┘
       │
       ▼  (POST http://localhost:8004/correct)
┌────────────────┐
│ Corrector Agent│  Returns Factually Corrected Text + Inline Citations
└────────────────┘
```

---

## ❓ FAQ & Troubleshooting

- **Q: What if `main` gets updated while I am working?**
  - Run `git fetch origin` followed by `git rebase origin/main` to get the latest updates without merge conflicts.
- **Q: Where do I find the Verifier Agent code reference?**
  - Check `agents/verifier_agent/` — it is fully implemented with 82 files serving as the reference standard for the project architecture.
- **Q: How do I test the full pipeline?**
  - Run Verifier Agent on port 8002 (`cd agents/verifier_agent && uvicorn api.main:app --port 8002`) and send HTTP POST requests to `http://localhost:8002/verify`.

---

**Happy Coding! Let's build the ultimate AI Trust Layer together! 🛡️**
