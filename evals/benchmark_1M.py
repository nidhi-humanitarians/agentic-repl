import json
import time
import sys
import re
import argparse
from pathlib import Path

# Add paths to sys.path so imports work seamlessly
repo_dir = Path(__file__).resolve().parent.parent
parent_dir = repo_dir.parent
for p in [str(parent_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from agentic_search_pi.agent.pi_loop import PiAgent
    from agentic_search_pi.config.settings import DEFAULT_MODEL
except ImportError:
    from agent.pi_loop import PiAgent
    from config.settings import DEFAULT_MODEL

# Book Title / Literary Source Metadata Mapping
BOOK_TITLES = {
    ("qa1", 0): "Winston Churchill, 'The Crisis' (Historical Fiction)",
    ("qa1", 1): "Herbert Vivian, 'Tunisia & Barbary Pirates' (Travelogue)",
    ("qa2", 0): "A. Barker, 'Manual of Surgical Operations' (Medical Treatise)",
    ("qa2", 1): "James Oliver Curwood, 'The Golden Snare' (Adventure Novel)",
    ("qa3", 0): "Sir Walter Scott, 'The Fortunes of Nigel' (Historical Romance)",
    ("qa3", 1): "Capt. John Smith / E. Arber, 'Colonial Works & Annals' (Historical)",
    ("qa4", 0): "Palmer Cox, 'The Brownies: Their Book' (Illustrated Verse)",
    ("qa4", 1): "P. G. Wodehouse, 'Something New' (Humorous Fiction)",
    ("qa5", 0): "Robert Herrick, 'The Common Lot' (American Fiction)",
    ("qa5", 1): "Frank L. Packard, 'The Miracle Man' (Crime Fiction)",
    ("qa6", 0): "Galena Historical Annals & Bench Street Records (History)",
    ("qa6", 1): "Benjamin Disraeli, 'Coningsby' (Political Fiction)",
    ("qa7", 0): "Historical Monograph on the United States Flag (History)",
    ("qa7", 1): "Goldwin Smith, 'Guesses at the Riddle of Existence' (Philosophy)",
    ("qa8", 0): "Anthony Hope, 'The Dolly Dialogues' (Society Comedy)",
    ("qa8", 1): "Victor Hugo, 'Les Misérables' (Classic Epic, trans.)",
    ("qa9", 0): "Clinical Pathology & Diseases of the Stomach (Medical)",
    ("qa9", 1): "UK Board of Trade Railway Inspection Reports (Government)",
    ("qa10", 0): "Richard Harding Davis, 'The King's Jackal' (Drama)",
    ("qa10", 1): "Arthur Symons, 'Collected Poems & Verse' (Literature)"
}

def check_answer_match(ground_truth: str, final_answer: str) -> bool:
    """Rigorous word-boundary evaluation to avoid digit/substring false matches."""
    final_lower = final_answer.lower()
    gt = ground_truth.strip().lower()

    if gt in ["yes", "no"]:
        return bool(re.search(rf"\b{gt}\b", final_lower))
    elif gt == "one":
        return bool(re.search(r"\b(one|1)\b", final_lower))
    elif gt in ["none", "nothing"]:
        # Check if the final answer or leading answer statement explicitly declares nothing/none/empty-handed
        first_line = final_lower.split("\n")[0]
        if any(term in first_line for term in ["nothing", "none", "empty-handed"]):
            return True
        if re.search(r"(carrying nothing|is empty-handed|has nothing)", final_lower):
            return True
        if re.search(r"answer:\s*(nothing|none|zero|empty-handed)", final_lower):
            return True
        return bool(re.search(r"\b(nothing|none|zero|empty-handed)\b", final_lower)) and not bool(re.search(r"is currently carrying", final_lower))
    else:
        return gt in final_lower

def run_qa_1M_multi_sample_benchmark(
    tasks=("qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"),
    samples=(0, 1),
    model=DEFAULT_MODEL,
    max_turns=15
):
    repo_dir = Path(__file__).resolve().parent.parent
    corpus_dir = repo_dir / "corpus"
    evals_dir = repo_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    summary_file = evals_dir / "babilong_1M_scorecard.json"

    print("=================================================================")
    print("   BABILONG 1M TOKEN (~2,500 PAGES) MULTI-TEXT BENCHMARK (2 TEXTS)")
    print("=================================================================")
    print(f"Model: {model}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Samples per Task: {samples}")
    print(f"Max Turns per Run: {max_turns}")
    print("=================================================================\n")

    new_results = []
    batch_start = time.time()

    for task in tasks:
        task_lower = task.lower().strip()
        for s_idx in samples:
            suffix = f"_s{s_idx}" if s_idx > 0 else ""
            txt_file = corpus_dir / f"babilong_{task_lower}_1M{suffix}.txt"
            meta_file = corpus_dir / f"babilong_{task_lower}_1M{suffix}_meta.json"

            if not txt_file.exists() or not meta_file.exists():
                print(f"Skipping {task} s{s_idx}: missing file {txt_file} or {meta_file}")
                continue

            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            question = meta["question"]
            ground_truth = meta["target"].strip().lower()
            approx_pages = meta.get("approx_pages", "~2,200")
            lines = meta.get("total_lines", 0)
            book_title = BOOK_TITLES.get((task_lower, s_idx), "Project Gutenberg PG-19 Classic")
            run_id = f"{task_lower.upper()}_s{s_idx}"

            print(f"\n-------------------------------------------------------------")
            print(f"Run:         {run_id} ({lines} lines, ~{approx_pages} pages)")
            print(f"Book Source: {book_title}")
            print(f"Question:    {question}")
            print(f"Target:      {ground_truth}")
            print(f"-------------------------------------------------------------")

            agent = PiAgent(
                corpus_path=str(txt_file),
                model=model,
                max_turns=max_turns
            )

            t0 = time.time()
            res = agent.search(query=question, verbose=False)
            elapsed = round(time.time() - t0, 2)

            final_ans = res.get("final_answer") or ""
            passed = check_answer_match(ground_truth, final_ans)
            status = "PASS" if passed else "FAIL"

            print(f"Status:      {status}")
            print(f"Turns Used:  {res.get('total_turns')}")
            print(f"REPL Calls:  {res.get('repl_executions')}")
            print(f"Latency:     {elapsed}s")
            print(f"Answer:      {final_ans.strip()[:140]}...")

            result_record = {
                "run_id": run_id,
                "task": task_lower.upper(),
                "sample_index": s_idx,
                "book_title": book_title,
                "status": status,
                "passed": passed,
                "approx_pages": approx_pages,
                "lines": lines,
                "question": question,
                "ground_truth": ground_truth,
                "final_answer": final_ans.strip(),
                "turns": res.get("total_turns"),
                "repl_executions": res.get("repl_executions"),
                "latency_seconds": elapsed,
                "trajectory": res.get("trajectory")
            }
            new_results.append(result_record)

            # INCREMENTAL SAVE TO SCORECARD (NEVER LOSE COMPLETED WORK)
            existing_runs_map = {}
            if summary_file.exists():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        prior_data = json.load(f)
                        for r in prior_data.get("results", []):
                            rid = r.get("run_id", f"{r.get('task', 'UNK')}_s0")
                            existing_runs_map[rid] = r
                except Exception:
                    pass
            existing_runs_map[run_id] = result_record

            def sort_key(item):
                rid = item.get("run_id", item.get("task", ""))
                m = re.match(r"QA(\d+)(?:_s(\d+))?", rid)
                if m:
                    task_num = int(m.group(1))
                    s_num = int(m.group(2)) if m.group(2) else 0
                    return (task_num, s_num)
                return (999, 0)

            cur_results = sorted(list(existing_runs_map.values()), key=sort_key)
            cur_passed = sum(1 for r in cur_results if r["passed"])
            cur_acc = (cur_passed / len(cur_results) * 100) if cur_results else 0

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark": "BABILong-1M-MultiText",
                    "model": model,
                    "overall_accuracy_percent": round(cur_acc, 1),
                    "passed_count": cur_passed,
                    "total_runs": len(cur_results),
                    "results": cur_results
                }, f, indent=2)

    batch_time = round(time.time() - batch_start, 2)

    # ACCUMULATE AND MERGE WITH EXISTING SCORECARD (NEVER OVERWRITE PRIOR ENTRIES)
    existing_runs_map = {}
    prior_total_time = 0.0

    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                prior_data = json.load(f)
                prior_total_time = prior_data.get("total_time_seconds", 0.0)
                for r in prior_data.get("results", []):
                    # Use run_id if available, else derive from task
                    rid = r.get("run_id", f"{r.get('task', 'UNK')}_s0")
                    existing_runs_map[rid] = r
        except Exception as e:
            print(f"Notice: reading existing scorecard: {e}")

    # Merge new results in
    for r in new_results:
        existing_runs_map[r["run_id"]] = r

    # Sort runs numerically by task then sample
    def sort_key(item):
        rid = item.get("run_id", item.get("task", ""))
        m = re.match(r"QA(\d+)(?:_s(\d+))?", rid)
        if m:
            task_num = int(m.group(1))
            s_num = int(m.group(2)) if m.group(2) else 0
            return (task_num, s_num)
        return (999, 0)

    combined_results = sorted(list(existing_runs_map.values()), key=sort_key)
    total_cumulative_time = round(prior_total_time + batch_time, 2)
    passed_count = sum(1 for r in combined_results if r["passed"])
    accuracy = (passed_count / len(combined_results) * 100) if combined_results else 0

    print("\n=================================================================")
    print("        CUMULATIVE MULTI-TEXT BENCHMARK SCORECARD (ALL RUNS)     ")
    print("=================================================================")
    print(f"{'Run ID':<8} | {'Status':<6} | {'Book Source':<32} | {'Turns':<6} | {'Time':<7}")
    print("-" * 68)
    for r in combined_results:
        bt = r.get("book_title", "PG-19 Novel")[:32]
        rid = r.get("run_id", r.get("task", ""))
        print(f"{rid:<8} | {r['status']:<6} | {bt:<32} | {r['turns']:<6} | {r['latency_seconds']:<6}s")
    print("-" * 68)
    print(f"Overall Cumulative Accuracy: {accuracy:.1f}% ({passed_count}/{len(combined_results)} passed)")
    print(f"Batch Time:                 {batch_time}s ({round(batch_time/60, 2)} minutes)")
    print(f"Total Cumulative Time:      {total_cumulative_time}s")
    print("=================================================================\n")

    scorecard_payload = {
        "benchmark": "BABILong-1M-MultiText",
        "model": model,
        "overall_accuracy_percent": accuracy,
        "passed_count": passed_count,
        "total_runs": len(combined_results),
        "total_time_seconds": total_cumulative_time,
        "results": combined_results
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(scorecard_payload, f, indent=2)

    print(f"Cumulative scorecard preserved and saved to: {summary_file}")
    return combined_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10")
    parser.add_argument("--samples", default="0,1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_turns", type=int, default=15)
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sample_list = [int(s.strip()) for s in args.samples.split(",") if s.strip()]
    run_qa_1M_multi_sample_benchmark(tasks=task_list, samples=sample_list, model=args.model, max_turns=args.max_turns)
