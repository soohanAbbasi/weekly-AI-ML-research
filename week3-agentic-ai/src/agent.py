"""
Week 3 — Minimal ReAct Agent From Scratch
==========================================
Architectural walkthrough of the ReAct loop + memory pipeline.
No LangChain, no AutoGen — just the raw mechanics.

Run:
    pip install numpy
    python agent.py
"""

import numpy as np


# =============================================================================
# MEMORY STORE
# =============================================================================

class MemoryStore:
    def __init__(self):
        self.facts = []

    def store(self, text):
        embedding = self._get_embedding(text)
        self.facts.append({"text": text, "embedding": embedding})

    def retrieve(self, query, top_k=3):
        if not self.facts:
            return []
        query_emb = self._get_embedding(query)
        scores = []
        for fact in self.facts:
            score = np.dot(query_emb, fact["embedding"])
            scores.append((score, fact["text"]))
        scores.sort(reverse=True)
        return [text for _, text in scores[:top_k]]

    def _get_embedding(self, text):
        # Mock embedding — replace with real model for semantic retrieval
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(128)
        return vec / np.linalg.norm(vec)

    def size(self):
        return len(self.facts)


# =============================================================================
# MOCK SEARCH DATABASE
# Replace _web_search() with Tavily or SerpAPI for real search
# =============================================================================

MOCK_DB = {
    "react agent": [
        "ReAct (Yao et al., ICLR 2023) improved HotpotQA by 12% over chain-of-thought by interleaving reasoning and tool use.",
        "ReAct loop: Thought -> Action -> Observation -> Thought. Every action has a traceable reasoning step.",
    ],
    "reflexion": [
        "Reflexion (Shinn et al., NeurIPS 2023) improves agents through verbal self-reflection stored in memory.",
        "Reflexion improved HumanEval pass@1 by ~10 percentage points over ReAct baseline without gradient updates.",
        "On AlfWorld, Reflexion reached 97% success after 3 reflection cycles vs 54% for ReAct baseline.",
    ],
    "memory agent": [
        "Effective agent memory needs 4 layers: in-context, external vector store, episodic, and procedural.",
        "Memory poisoning — storing wrong facts early — is one of the most common production failure modes.",
    ],
    "multi agent": [
        "AutoGen (Microsoft 2023) models agents as conversational participants.",
        "LangGraph models workflows as directed graphs — most debuggable approach for production systems.",
    ]
}


# =============================================================================
# TOOLS
# =============================================================================

def _memory_retrieve(query, memory):
    results = memory.retrieve(query, top_k=3)
    if not results:
        return "No memories found."
    return "\n".join(f"[Memory {i+1}] {r}" for i, r in enumerate(results))

def _web_search(query):
    query_lower = query.lower()
    for key, results in MOCK_DB.items():
        if any(word in query_lower for word in key.split()):
            return "\n".join(f"[Result {i+1}] {r}" for i, r in enumerate(results[:2]))
    return "No results found."

def _memory_store(fact, memory):
    memory.store(fact)
    return f"Saved to memory. Total facts: {memory.size()}"

TOOLS = {
    "memory_retrieve": lambda query, memory: _memory_retrieve(query, memory),
    "web_search":      lambda query, memory: _web_search(query),
    "memory_store":    lambda fact,  memory: _memory_store(fact, memory),
    "final_answer":    lambda answer, memory: answer,
}


# =============================================================================
# AGENT BRAIN (mock)
# In production: replace with Claude API call (tool_use mode)
# =============================================================================

def agent_think(query, memory_results, search_results, step):
    if step == 1:
        return {
            "thought": f"I need to answer: '{query}'. Let me check memory first.",
            "tool": "memory_retrieve",
            "input": query
        }
    if step == 2 and "No memories found" in memory_results:
        return {
            "thought": "Memory is empty. I need to search the web.",
            "tool": "web_search",
            "input": query
        }
    if step == 2 and "Memory" in memory_results:
        return {
            "thought": "Found relevant facts in memory. Enough to answer.",
            "tool": "final_answer",
            "input": f"Based on memory: {memory_results}"
        }
    if step == 3:
        first_result = search_results.split('\n')[0].replace('[Result 1] ', '')
        return {
            "thought": "Search results found. Saving key fact to memory.",
            "tool": "memory_store",
            "input": first_result
        }
    if step == 4:
        return {
            "thought": "Fact saved. Enough information to answer.",
            "tool": "final_answer",
            "input": search_results
        }
    return None


# =============================================================================
# AGENT LOOP
# =============================================================================

def run_agent(query, memory):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"Memory at start: {memory.size()} facts")
    print(f"{'='*60}")

    memory_results = ""
    search_results = ""

    for step in range(1, 6):
        decision = agent_think(query, memory_results, search_results, step)
        if decision is None:
            break

        print(f"\n[Step {step}]")
        print(f"  Thought  : {decision['thought']}")
        inp = decision['input']
        print(f"  Action   : {decision['tool']}('{inp[:60]}{'...' if len(inp) > 60 else ''}')")

        result = TOOLS[decision['tool']](decision['input'], memory)
        print(f"  Observe  : {str(result)[:120]}")

        if decision['tool'] == 'memory_retrieve':
            memory_results = result
        elif decision['tool'] == 'web_search':
            search_results = result
        elif decision['tool'] == 'final_answer':
            print(f"\n{'─'*60}")
            print(f"ANSWER: {str(result)[:300]}")
            print(f"Memory after: {memory.size()} facts")
            return result

    return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    shared_memory = MemoryStore()

    print("EXPERIMENT: Two queries sharing one memory store")
    print("Query 1 — cold start (empty memory)")
    print("Query 2 — warm start (benefits from Query 1)\n")

    run_agent("What is the ReAct framework and how does it work?", shared_memory)

    print(f"\n{'='*60}")
    print(f"Memory now has {shared_memory.size()} fact(s) stored from Query 1")
    print(f"{'='*60}")

    run_agent("How does Reflexion improve on ReAct for language agents?", shared_memory)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Query 1 — 4 steps (cold start, had to search)")
    print("Query 2 — 2 steps (warm start, memory hit)")
    print(f"Final memory size: {shared_memory.size()} facts")
