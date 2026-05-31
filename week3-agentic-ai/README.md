# Week 3 — Agentic AI: Architecture of Autonomous Systems

Part of the **Weekly AI/ML Research Series** (12 weeks).

This repository contains the experiment code, figures, and notebook for Week 3 of the series.

**Blog post:** [Agentic AI: Architecture of Autonomous Systems](#)  
**LinkedIn:** [Soohan Abbasi](https://www.linkedin.com/in/soohan-abbasi-36267b183/)

---

## What This Experiment Does

Builds a minimal ReAct agent **from scratch** using only the Anthropic API — no LangChain, no AutoGen.

The goal is not to produce novel results. It is to make the architecture visible: every tool call, every reasoning step, every memory retrieval — all explicit and traceable.

Two queries share one memory store:
- **Query 1** runs cold (empty memory) — must search, store, then answer. Takes 4 steps.
- **Query 2** runs warm — finds facts stored by Query 1, skips web search entirely. Takes 2 steps.

This demonstrates the core value of persistent memory in agentic systems.

---

## Repo Structure

```
week3-agentic-ai/
│
├── notebooks/
│   └── week3_experiment.ipynb     # Full Kaggle notebook (all 7 steps)
│
├── src/
│   └── agent.py                   # Clean standalone version of the agent
│
├── figures/
│   ├── react_loop_diagram.png     # Figure 1 — ReAct loop diagram
│   ├── reflexion_vs_react.png     # Figure 2 — Benchmark comparison chart
│   └── experiment_trace.png       # Figure 3 — Agent trace (cold vs warm)
│
└── README.md
```

---

## Key Components

### MemoryStore
Simple cosine-similarity vector store. Facts stored as 128-dim embeddings, retrieved by similarity to the query.

> Limitation: uses mock embeddings (seeded random vectors). Not semantically meaningful — retrieval works on keyword overlap, not true semantic similarity. In production: replace with `text-embedding-3-small` or similar.

### Tools
Four tools the agent can call:
- `memory_retrieve` — check memory before searching
- `web_search` — mock search database (replace with Tavily/SerpAPI in production)
- `memory_store` — save a fact for future queries
- `final_answer` — return the answer

### Agent Loop
Core ReAct loop: Thought → Action → Observation → repeat until `final_answer`.

---

## How to Run

### On Kaggle (recommended)
Open `notebooks/week3_experiment.ipynb` directly on Kaggle. No API key needed — runs in mock mode.

### Locally
```bash
pip install anthropic numpy matplotlib
python src/agent.py
```

> Note: mock mode does not require an API key. The agent brain is hardcoded to demonstrate the ReAct loop without a live API call.

---

## What I Learned Building Without a Framework

Two things became clear that papers alone do not make obvious:

**Tool description quality matters more than expected.** A vague description causes the model to call `web_search` when `memory_retrieve` was correct — purely because the priority was not explicit. Frameworks handle this with opinionated defaults. When those defaults are wrong, you cannot see why.

**Mock embeddings are brittle.** Retrieval works when surface-level keyword overlap is high. It fails silently on semantically similar queries with different wording. A framework abstracts this. Building without one makes the failure immediately visible.

---

## Series

| Week | Topic |
|---|---|
| 1 | Chain-of-Thought and Beyond: How LLMs Actually Learn to Reason |
| 2 | Small Language Models |
| **3** | **Agentic AI: Architecture of Autonomous Systems** |
| 4 | Fine-Tuning vs Prompting: When Do You Actually Need to Train? |
| ... | ... |

---

## References

1. Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
2. Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
3. Wu et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.
4. Schick et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*.
