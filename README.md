# Agentic Search via Code REPL (Pi Agent)

> **Research Question:** Can an agent navigate, search, and reason across massive, unstructured long-context documents (2,000+ page books, insurance binders, clinical trial dossiers) by writing and executing code in an interactive REPL, rather than relying on passive context-window stuffing?

---

## 1. Executive Summary

Standard Large Language Models struggle with documents exceeding thousands of pages:
1. **Passive Context-Stuffing Fails:** Cramming millions of tokens into a single prompt causes severe **attention dilution** ("lost in the middle"). On multi-hop reasoning tasks (like BABILong QA2/QA3), standard model accuracy drops to near 0%.
2. **High Latency & Quadratic Cost:** Processing 1,000,000 tokens per query costs \$5–\$15 and takes minutes per forward pass.
3. **No Verifiability:** Black-box outputs make it impossible to audit whether the model reasoned or guessed.

### The Solution: Active Programmatic Search via Python REPL
Instead of treating the book as a prompt, this system treats the book as an **interactive file environment**:
* The **Pi Agent** writes Python scripts (`grep`, regex, line slicers, sorting algorithms) inside a persistent **Python REPL**.
* The agent only pulls small, relevant 10–20 line windows into context.
* **Results:** **79.0% accuracy across 100 distinct 2,500-page books** (~240,000 cumulative pages), average query latency of **~25.8 seconds**, and a **99.6% token cost reduction** compared to raw context stuffing.

---

## 2. Repository Architecture

```text
agentic-repl/
├── README.md                      # Complete project documentation, benchmark records & analysis
├── requirements.txt               # Dependencies (openai, requests, pydantic, python-dotenv)
├── .gitignore                     # Strictly excludes all data/corpus files, cache, and secrets
├── .env.example                   # Configuration template for Particle AI credentials
├── run_search.py                  # CLI entry point to run interactive or benchmark queries
│
├── config/                        # Model & API endpoint configurations
│   ├── settings.py                # Particle AI base URL, API keys, and model mappings
│   └── __init__.py
│
├── agent/                         # The Pi Agent reasoning loop & prompt harness
│   ├── client.py                  # OpenAI-compatible Particle AI client wrapper
│   ├── pi_loop.py                 # Multi-turn execution loop with tool dispatch & fallback
│   ├── prompts.py                 # System search instructions & chronological sorting heuristics
│   ├── tools_schema.py            # Function calling schema for the python_repl tool
│   └── __init__.py
│
├── repl/                          # The persistent Python execution sandbox
│   ├── session.py                 # Persistent Python session with output truncation bounds
│   ├── helpers.py                 # Injected utilities (grep, read_lines, file_info)
│   └── __init__.py
│
├── corpus/                        # Document loaders & dataset streaming
│   ├── download_babilong.py       # Single-sample streaming downloader
│   └── download_batch.py          # High-speed batch streaming downloader for 1M datasets
│
└── evals/                         # Benchmark runners & evaluation scorecards
    ├── evaluator.py               # Canonical semantic answer evaluator (quantities, binary, entities)
    ├── benchmark_1M.py            # Multi-text runner with scorecard accumulation
    └── run_100_benchmark.py       # Parallel multi-worker 100-sample benchmark runner
```

> **Data Policy:** In accordance with repository best practices, raw corpus texts and datasets are **not tracked in Git** and are excluded via `.gitignore`. The dataset streamers (`corpus/download_batch.py`) fetch target corpora on demand.

---

## 3. Quickstart & Installation

### A. Clone and Install Dependencies
```bash
git clone https://github.com/nidhi-humanitarians/agentic-repl.git
cd agentic-repl
pip install -r requirements.txt
```

### B. Configure Particle AI Credentials
Copy `.env.example` to `.env` and insert your Particle AI API key:
```bash
cp .env.example .env
```
Supported models on Particle AI:
* `deepseek-v4.1-flash` (Default — fast, strong code generation and reasoning)
* `glm-5.3-flash` (Alternative evaluation backbone)

---

## 4. Usage

### A. Run an Interactive Search
Run a search query against any text file:
```bash
python -m run_search --corpus path/to/your_book.txt --query "Where did the author define Hamiltonian?"
```

### B. Download 1M Token Samples On Demand
Download samples from the official BABILong benchmark:
```bash
# Download 10 samples (s0-s9) for tasks QA1 through QA10 at the 1M token (~2,500 pages) scale:
python corpus/download_batch.py --tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --count 10
```

### C. Run the Parallel 100-Book Benchmark
Benchmark all 10 tasks across 100 distinct 2,500-page books using parallel workers:
```bash
python evals/run_100_benchmark.py --tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --count 10 --workers 5
```

---

## 5. Complete Empirical Benchmark Records (100 Books / ~240,000 Pages at 1M Scale)

Evaluated across **100 distinct, full-length 2,000+ page books** from Project Gutenberg PG-19 (10 samples per task across QA1 through QA10):

| Task | Reasoning Category | Completed | Passed | Accuracy (%) | Avg Turns | Avg Latency | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **QA1** | Single Supporting Fact (1-Hop Lookup) | 10 | 10 | **100.0%** | 12.3 | 20.5s | **PERFECT** |
| **QA2** | Two Supporting Facts (2-Hop State Tracking) | 10 | 3 | **30.0%** | 15.0 | 32.3s | Bottleneck |
| **QA3** | Three Supporting Facts (3-Hop Object Transfer) | 10 | 2 | **20.0%** | 15.0 | 36.4s | Bottleneck |
| **QA4** | Two Argument Relations (Spatial Deduction) | 10 | 10 | **100.0%** | 14.7 | 44.8s | **PERFECT** |
| **QA5** | Three Argument Relations (Actor Interactions) | 10 | 9 | **90.0%** | 11.5 | 17.3s | **STRONG** |
| **QA6** | Yes/No Fact Verification (Binary Status) | 10 | 10 | **100.0%** | 13.6 | 24.5s | **PERFECT** |
| **QA7** | Counting Aggregation (Net Held Inventory) | 10 | 7 | **70.0%** | 13.3 | 30.6s | Moderate |
| **QA8** | Lists & Held Inventory (Active Set) | 10 | 9 | **90.0%** | 10.6 | 14.6s | **STRONG** |
| **QA9** | Simple Negation (Negative Logic) | 10 | 10 | **100.0%** | 11.9 | 21.4s | **PERFECT** |
| **QA10** | Indefinite Knowledge (Disjunction / Either-Or) | 10 | 9 | **90.0%** | 9.1 | 12.3s | **STRONG** |
| **TOTAL** | **Full 10-Task Benchmark Summary** | **100** | **79** | **79.0%** | **12.7** | **25.8s** | **ROBUST BASELINE** |

* Cumulative scorecard log with full trajectories is preserved in `evals/babilong_100_scorecard.json`.

---

## 6. Comprehensive Failure Taxonomy & Root Cause Analysis

Across the 100 evaluated books, **79 passed and 21 failed**. A deep-dive analysis of all 21 failure trajectories reveals four distinct failure categories:

```
                          DISTRIBUTION OF THE 21 FAILURES
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ██████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░  │
  │ Multi-Hop State Tracking & Turn Exhaustion (QA2 & QA3): 15 (71.4%)     │
  │ Counting Aggregation in Dense Noise (QA7):               3 (14.3%)     │
  │ Inventory Multi-Item Drop (QA8):                         1 (4.8%)      │
  │ Actor Transfer Ambiguity (QA5):                          1 (4.8%)      │
  │ Uncertainty Phrasing (QA10):                             1 (4.8%)      │
  └────────────────────────────────────────────────────────────────────────┘
```

### Category 1: Multi-Hop State Tracking & Turn Budget Exhaustion (15 Failures — 71.4% of all errors)
* **Tasks Affected:** QA2 (7 failures) and QA3 (8 failures).
* **The Mechanism:** 
  * In QA3, questions ask for previous states: *"Where was the apple before the bedroom?"*.
  * The agent spends 10–12 turns searching for current character positions or issuing broad regex queries (`grep("John|Mary|Sandra|Daniel")`) that return 200+ matches.
  * Sifting through 200 matches causes the agent to hit the 15-turn ceiling before it can synthesize the reverse chronological order.
  * In QA2, the agent occasionally stops at an intermediate location or anchors on background narrative geography (e.g. *"Simon Glover's house in Perth"* or *"Atlanta"*).

### Category 2: Counting Aggregation in Dense Narrative Text (3 Failures — 14.3%)
* **Task Affected:** QA7 (`QA7_s1`, `QA7_s2`, `QA7_s3`).
* **The Mechanism:** QA7 requires computing net inventory ($\text{pickups} - \text{drops}$) across 55,000 lines. In noisy texts with character homonyms, the agent missed one drop event or misattributed a common noun mentioned in background dialogue.

### Category 3: Multiple Item Inventory Capture (1 Failure — 4.8%)
* **Task Affected:** QA8 (`QA8_s7`).
* **The Mechanism:** Question asked: *"What is John carrying?"*. Target was `apple,football`. The agent successfully found the apple acquisition but stopped before checking if John had also acquired the football, answering only `The apple`.

### Category 4: Actor & Uncertainty Phrasing (2 Failures — 9.5%)
* **Tasks Affected:** QA5 (`QA5_s2`) and QA10 (`QA10_s3`).
* **The Mechanism:** In QA10_s3 (Target: `maybe`), the model correctly deduced that Bill was either in the school or office, but explained it descriptively without explicitly uttering the word `maybe`.

---

## 7. Proposed Systematic Solutions (The 4-Layer Harness Architecture)

To resolve the 21 failure cases systematically without brittle prompt engineering, the following four architectural layers are being deployed:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Tool-Boundary Middleware (Defensive Protocol Sanitizers) │
├─────────────────────────────────────────────────────────────┤
│ 2. Out-of-Band Steering Controller (Pathology Detectors)   │
├─────────────────────────────────────────────────────────────┤
│ 3. Typed Final Output Contracts (Schema Bounding)           │
├─────────────────────────────────────────────────────────────┤
│ 4. Dynamic Turn Allocation (Task-Adaptive Budgets)          │
└─────────────────────────────────────────────────────────────┘
```

1. **Tool-Boundary Middleware:**
   * **Protocol Auto-Healing:** Automatically intercepts and evaluates raw `<｜DSML｜ ...>` and XML tool tokens before they enter message history.
   * **Query Auto-Faceting:** When `grep` matches > 40 lines, the REPL returns a quartile histogram (e.g. *"210 matches clustered between lines 15,000–30,000"*), preventing search explosion.
2. **Out-of-Band Steering Controller:**
   * Monitors trajectory invariants after each turn. If an agent loops on broad queries or reads background narrative lore for 2 consecutive turns, it injects a steering nudge: *"Focus strictly on synthetic movement verbs: journeyed to, went to, moved to."*
3. **Task-Adaptive Dynamic Turn Budgeting:**
   * Allocate 5–8 turns for single-hop and spatial queries (QA1, QA4, QA9), and grant 20–25 turns for 3-hop transfer queries (QA2, QA3), eliminating turn-ceiling truncation.
4. **Typed Output Contracts & Canonical Normalization:**
   * Constrain final answers to a structured Pydantic schema with canonical synonym mapping (`0` $\leftrightarrow$ `zero` $\leftrightarrow$ `none` $\leftrightarrow$ `nothing`, `either/or` $\leftrightarrow$ `maybe`).

---

## 8. License & Acknowledgments

* **Agent Loop & Harness:** Inspired by Pi Agent architecture ([Earendil](https://earendil.com)).
* **Inference Engine:** Powered by [Particle AI](https://particle.ai).
* **Benchmark & Texts:** BABILong ([AIRI / DeepPavlov.ai / LIMS](https://github.com/booydar/babilong)) and Project Gutenberg PG-19 (DeepMind).
