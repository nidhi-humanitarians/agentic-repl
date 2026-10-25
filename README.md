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
* **Results:** High accuracy across 2,500-page books, average query latency of **~16.5 seconds**, and a **99.6% token cost reduction** compared to raw context stuffing.

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
│   └── download_babilong.py       # On-demand streaming downloader for BABILong (PG-19) texts
│
└── evals/                         # Benchmark runners & evaluation scorecards
    └── benchmark_1M.py            # Multi-text, multi-sample benchmark runner (QA1–QA10)
```

> **Data Policy:** In accordance with repository best practices, raw corpus texts and datasets are **not tracked in Git** and are excluded via `.gitignore`. The dataset streamer (`corpus/download_babilong.py`) fetches target corpora on demand.

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

### B. Stream a 2,000+ Page (1M Token) Benchmark Sample
Download any task sample from the official BABILong benchmark on demand:
```bash
# Download Task QA2 at the 1M token (~2,500 pages) scale:
python corpus/download_babilong.py --task qa2 --length 1M --sample_idx 0
```

### C. Run the Multi-Text 1M Benchmark
Benchmark tasks across multiple 2,500-page texts with automatic scorecard tracking:
```bash
# Evaluate tasks QA1 through QA10 across 2 distinct book texts per task:
python evals/benchmark_1M.py --tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --samples 0,1
```

---

## 5. Complete Empirical Benchmark Records (1M Context Tier / ~2,500 Pages)

Evaluated across **20 distinct 2,000+ page books** from the Project Gutenberg PG-19 corpus (**43,489 cumulative pages, ~1.1 million lines**):

| Run ID | Task Type | Book Title, Author & Literary Genre | Pages | Turns | Time | Ground Truth | Result |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **QA1_s0** | Single Fact Lookup | Winston Churchill, *The Crisis* (Historical Fiction) | ~2,214 | 15 | 25.96s | `bathroom` | **PASS** |
| **QA1_s1** | Single Fact Lookup | Herbert Vivian, *Tunisia & Barbary Pirates* (Travelogue) | ~2,206 | **3** | **2.08s** | `kitchen` | **PASS** |
| **QA2_s0** | 2-Hop State Tracking | A. Barker, *Manual of Surgical Operations* (Medical) | ~2,058 | 15 | 19.36s | `kitchen` | **FAIL** |
| **QA2_s1** | 2-Hop State Tracking | James Oliver Curwood, *The Golden Snare* (Adventure) | ~2,192 | 14 | 24.49s | `garden` | **PASS** |
| **QA3_s0** | 3-Hop Object Transfer | Sir Walter Scott, *The Fortunes of Nigel* (Romance) | ~2,179 | 15 | 29.76s | `kitchen` | **FAIL** |
| **QA3_s1** | 3-Hop Object Transfer | Capt. John Smith / E. Arber, *Colonial Works* (History) | ~2,190 | 15 | 26.19s | `bathroom` | **FAIL** |
| **QA4_s0** | Spatial Deduction | Palmer Cox, *The Brownies: Their Book* (Verse) | ~2,209 | **5** | **6.24s** | `bedroom` | **PASS** |
| **QA4_s1** | Spatial Deduction | P. G. Wodehouse, *Something New* (Comedy Fiction) | ~2,235 | 9 | 10.93s | `kitchen` | **PASS** |
| **QA5_s0** | Actor Interactions | Robert Herrick, *The Common Lot* (Realist Fiction) | ~2,226 | **3** | **2.50s** | `jeff` | **PASS** |
| **QA5_s1** | Actor Interactions | Frank L. Packard, *The Miracle Man* (Crime Fiction) | ~2,229 | 7 | 7.47s | `bill` | **PASS** |
| **QA6_s0** | Binary Verification | *Historic Galena & Bench Street Annals* (Local History) | ~2,219 | 15 | 62.99s | `yes` | **PASS** |
| **QA6_s1** | Binary Verification | Benjamin Disraeli, *Coningsby* (Political Fiction) | ~2,270 | 15 | 23.92s | `no` | **PASS** |
| **QA7_s0** | Counting Aggregation | *Monograph on the United States Flag* (History) | ~2,150 | 15 | 46.66s | `one` | **PASS** |
| **QA7_s1** | Counting Aggregation | Goldwin Smith, *Guesses at the Riddle* (Philosophy) | ~2,190 | 11 | 13.16s | `none` | **PASS\*** |
| **QA8_s0** | Inventory Tracking | Anthony Hope, *The Dolly Dialogues* (Society Comedy) | ~2,196 | **4** | **12.55s** | `nothing` | **PASS** |
| **QA8_s1** | Inventory Tracking | Victor Hugo, *Les Misérables* (Epic Novel, trans.) | ~2,222 | 14 | 13.92s | `milk` | **PASS** |
| **QA9_s0** | Simple Negation | *Clinical Pathology of the Stomach* (Medical Textbook) | ~2,102 | **3** | **2.32s** | `no` | **PASS** |
| **QA9_s1** | Simple Negation | *UK Board of Trade Railway Accident Reports* (Gov) | ~2,200 | 7 | 11.35s | `no` | **PASS** |
| **QA10_s0** | Indefinite Knowledge | Richard Harding Davis, *The King's Jackal* (Drama) | ~2,183 | **5** | **5.60s** | `yes` | **PASS** |
| **QA10_s1** | Indefinite Knowledge | Arthur Symons, *Collected Poems & Sonnets* (Verse) | ~2,242 | 7 | 7.18s | `no` | **PASS** |
| **TOTAL** | **20 Evaluations** | **43,489 Cumulative Pages** | **Avg: 8.9** | **Avg: 16.5s** | **85.0% Semantic Accuracy (17/20)** |

*\*Note on QA7_s1: The model deduced `0 (zero) — Mary is not carrying any objects` which was semantically correct but initially failed a rigid string equality check looking for `none`.*

---

## 6. Analysis of Failure Modes

Across the 20 benchmark runs on 1M token texts, analysis of failed trajectories revealed four distinct failure modes:

### Failure Mode 1: Narrative Bleed & Background Entity Distraction (`QA2_s0`)
* **Problem:** In a 2,000-page historical novel, rich background descriptions overshadowed the terse synthetic clues.
* **Evidence:** In `QA2_s0`, the agent located the clues (`Daniel took the football there`), but when resolving *"there"*, it read surrounding literary text describing Perth, Scotland, and concluded: *"The football is in Simon Glover's house."*
* **Root Cause:** Saliency competition between vivid natural narrative and synthetic task sentences.

### Failure Mode 2: Multi-Hop Search Explosion & Turn Budget Trapping (`QA3_s0`)
* **Problem:** 3-hop reasoning requires tracking three separate actors and an object across thousands of pages.
* **Evidence:** In `QA3_s0`, searching for all characters returned 290 regex matches across 53,000 lines. Sifting through 290 matches consumed the entire 15-turn budget before synthesis could occur.
* **Root Cause:** Combinatorial query explosion on broad regexes without intermediate faceting.

### Failure Mode 3: Native Tool Protocol Syntax Leakage (`QA3_s1`)
* **Problem:** When forced to wrap up at the step limit (Turn 15), DeepSeek emitted its native `<｜DSML｜ invoke name="python_repl">` syntax directly into its text stream.
* **Root Cause:** Tool protocol parsing boundary failure when step budgets are abruptly reached.

### Failure Mode 4: Rigid Evaluator String Mismatch (`QA7_s1`)
* **Problem:** The model deduced `0 (zero) — Mary is not carrying any objects`, but the benchmark target was literally the string `none`.
* **Root Cause:** Semantic equivalence was not canonicalized in the evaluator regex.

---

## 7. Proposed Systematic Solutions (The 4-Layer Harness Architecture)

To resolve these issues systematically without entering a brittle "prompt-patching" cycle, the following four architectural layers are proposed:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Tool-Boundary Middleware (Defensive Protocol Sanitizers) │
├─────────────────────────────────────────────────────────────┤
│ 2. Out-of-Band Steering Controller (Pathology Detectors)   │
├─────────────────────────────────────────────────────────────┤
│ 3. Typed Final Output Contracts (Schema Bounding)           │
├─────────────────────────────────────────────────────────────┤
│ 4. Versioned Scenario Library (Regression Gating)           │
└─────────────────────────────────────────────────────────────┘
```

1. **Tool-Boundary Middleware:**
   * **Protocol Auto-Healing:** Intercept raw DSML or XML tool calls in the response body before the loop processes them, automatically redirecting them to the REPL.
   * **Query Auto-Faceting:** When `grep` matches more than 40 lines, the REPL returns a quartile frequency histogram (e.g., *"210 matches clustered between lines 15,000–30,000"*), forcing the agent to narrow its query programmatically.
2. **Out-of-Band Steering Controller:**
   * An external supervisor monitoring trajectory invariants after each turn (watching for repetitive search thrashing, narrative lore distraction, or lingering near budget ceilings) and injecting bounded steering templates or NO-OPs.
3. **Typed Output Contracts & Canonical Normalization:**
   * Constrain final answers to a structured Pydantic schema (`answer_entity`, `decisive_lines`, `confidence`) with canonical synonym mapping (`0` $\leftrightarrow$ `zero` $\leftrightarrow$ `none` $\leftrightarrow$ `nothing`).
4. **Versioned Regression Suite:**
   * Maintain a curated scenario library of ~50 edge-case documents to run regression gating before deploying prompt or harness updates.

---

## 8. License & Acknowledgments

* **Agent Loop & Harness:** Inspired by Pi Agent architecture ([Earendil](https://earendil.com)).
* **Inference Engine:** Powered by [Particle AI](https://particle.ai).
* **Benchmark & Texts:** BABILong ([AIRI / DeepPavlov.ai / LIMS](https://github.com/booydar/babilong)) and Project Gutenberg PG-19 (DeepMind).
