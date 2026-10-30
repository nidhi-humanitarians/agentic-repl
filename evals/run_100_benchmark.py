import json
import time
import sys
import re
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths to sys.path
repo_dir = Path(__file__).resolve().parent.parent
parent_dir = repo_dir.parent
for p in [str(parent_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from agentic_search_pi.agent.pi_loop import PiAgent
    from agentic_search_pi.config.settings import DEFAULT_MODEL
    from agentic_search_pi.evals.evaluator import check_answer_match
except ImportError:
    from agent.pi_loop import PiAgent
    from config.settings import DEFAULT_MODEL
    from evals.evaluator import check_answer_match

# File lock for thread-safe incremental scorecard writing
file_lock = threading.Lock()

def evaluate_single_sample(run_spec: dict, model: str = DEFAULT_MODEL, max_turns: int = 15) -> dict:
    """Evaluates a single 2,000+ page book sample in an isolated thread."""
    task = run_spec["task"]
    s_idx = run_spec["sample_index"]
    run_id = run_spec["run_id"]
    txt_path = run_spec["txt_path"]
    meta_path = run_spec["meta_path"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    question = meta["question"]
    ground_truth = meta["target"].strip().lower()
    approx_pages = meta.get("approx_pages", 2200)
    lines = meta.get("total_lines", 0)

    agent = PiAgent(
        corpus_path=str(txt_path),
        model=model,
        max_turns=max_turns
    )

    t0 = time.time()
    res = agent.search(query=question, verbose=False)
    elapsed = round(time.time() - t0, 2)

    final_ans = (res.get("final_answer") or "").strip()
    eval_res = check_answer_match(ground_truth=ground_truth, final_answer=final_ans)
    passed = eval_res["passed"]
    status = "PASS" if passed else "FAIL"

    return {
        "run_id": run_id,
        "task": task.upper(),
        "sample_index": s_idx,
        "status": status,
        "passed": passed,
        "approx_pages": approx_pages,
        "lines": lines,
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": final_ans,
        "eval_details": eval_res,
        "turns": res.get("total_turns"),
        "repl_executions": res.get("repl_executions"),
        "latency_seconds": elapsed,
        "trajectory": res.get("trajectory")
    }

def run_100_benchmark(
    tasks=("qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"),
    samples_per_task: int = 10,
    max_workers: int = 5,
    model: str = DEFAULT_MODEL,
    max_turns: int = 15,
    resume: bool = True
):
    repo_dir = Path(__file__).resolve().parent.parent
    corpus_dir = repo_dir / "corpus"
    evals_dir = repo_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    summary_file = evals_dir / "babilong_100_scorecard.json"

    print("=================================================================")
    print("   BABILONG 1M (2,000+ PAGES) BENCHMARK: 100 BOOKS (QA1 - QA10)  ")
    print("=================================================================")
    print(f"Model: {model}")
    print(f"Tasks: {', '.join(tasks)} ({len(tasks)} tasks)")
    print(f"Samples per Task: {samples_per_task} (Total runs: {len(tasks) * samples_per_task})")
    print(f"Parallel Workers: {max_workers}")
    print(f"Max Turns per Run: {max_turns}")
    print("=================================================================\n")

    # Load existing runs if resuming
    existing_runs = {}
    if summary_file.exists() and resume:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                prior = json.load(f)
                for r in prior.get("results", []):
                    existing_runs[r["run_id"]] = r
            print(f"Resuming from existing scorecard: {len(existing_runs)} runs already completed.\n")
        except Exception as e:
            print(f"Notice: starting fresh scorecard ({e})")

    # Enumerate all target run specifications
    run_specs = []
    for task in tasks:
        t_clean = task.lower().strip()
        for s_idx in range(samples_per_task):
            run_id = f"{t_clean.upper()}_s{s_idx}"
            if resume and run_id in existing_runs:
                continue

            txt_file = corpus_dir / f"babilong_{t_clean}_1M_s{s_idx}.txt"
            meta_file = corpus_dir / f"babilong_{t_clean}_1M_s{s_idx}_meta.json"
            
            # Fallback for s0 naming convention
            if s_idx == 0 and not txt_file.exists():
                txt_file = corpus_dir / f"babilong_{t_clean}_1M.txt"
                meta_file = corpus_dir / f"babilong_{t_clean}_1M_meta.json"

            if not txt_file.exists() or not meta_file.exists():
                print(f"Warning: missing files for {run_id} ({txt_file.name}), skipping.")
                continue

            run_specs.append({
                "task": t_clean,
                "sample_index": s_idx,
                "run_id": run_id,
                "txt_path": txt_file,
                "meta_path": meta_file
            })

    total_to_run = len(run_specs)
    print(f"Queued {total_to_run} pending runs across {max_workers} worker threads.\n")
    if total_to_run == 0:
        print("All runs already completed in scorecard!")
        return existing_runs

    start_time = time.time()
    completed_in_batch = 0

    def update_scorecard_file(record: dict):
        with file_lock:
            existing_runs[record["run_id"]] = record
            
            def sort_key(item):
                rid = item.get("run_id", "")
                m = re.match(r"QA(\d+)_s(\d+)", rid)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                return (999, 0)

            all_results = sorted(list(existing_runs.values()), key=sort_key)
            passed = sum(1 for r in all_results if r["passed"])
            acc = round((passed / len(all_results)) * 100, 1) if all_results else 0

            with open(summary_file, "w", encoding="utf-8") as sf:
                json.dump({
                    "benchmark": "BABILong-1M-100Books",
                    "model": model,
                    "overall_accuracy_percent": acc,
                    "passed_count": passed,
                    "total_runs": len(all_results),
                    "results": all_results
                }, sf, indent=2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_sample, spec, model, max_turns): spec for spec in run_specs}

        for future in as_completed(futures):
            spec = futures[future]
            completed_in_batch += 1
            try:
                res = future.result()
                update_scorecard_file(res)
                q_snippet = res['question'][:30]
                ans_snippet = res['final_answer'].replace('\n', ' ')[:40]
                print(f"[{completed_in_batch}/{total_to_run}] {res['run_id']:<8} -> {res['status']:<4} ({res['latency_seconds']}s, {res['turns']} turns) | Q: '{q_snippet}' -> '{ans_snippet}'")
            except Exception as e:
                print(f"[{completed_in_batch}/{total_to_run}] {spec['run_id']:<8} -> ERROR: {e}")

    total_batch_time = round(time.time() - start_time, 2)
    print("\n=================================================================")
    print("                    100-BOOK BENCHMARK COMPLETED                 ")
    print("=================================================================")
    print(f"Batch Time: {total_batch_time}s ({round(total_batch_time/60, 2)} minutes)")

    # Print final summary table
    if not summary_file.exists():
        print(f"Warning: scorecard file {summary_file} does not exist yet.")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    results = final_data.get("results", [])
    task_stats = {}
    for r in results:
        t = r["task"]
        if t not in task_stats:
            task_stats[t] = {"total": 0, "passed": 0, "times": [], "turns": []}
        task_stats[t]["total"] += 1
        if r["passed"]:
            task_stats[t]["passed"] += 1
        task_stats[t]["times"].append(r["latency_seconds"])
        task_stats[t]["turns"].append(r["turns"])

    print(f"\n{'Task':<8} | {'Accuracy':<10} | {'Passed':<8} | {'Avg Turns':<10} | {'Avg Latency':<12}")
    print("-" * 55)
    for t in sorted(task_stats.keys(), key=lambda x: int(x.replace("QA", ""))):
        st = task_stats[t]
        acc_pct = round((st["passed"] / st["total"]) * 100, 1)
        avg_time = round(sum(st["times"]) / len(st["times"]), 1)
        avg_turns = round(sum(st["turns"]) / len(st["turns"]), 1)
        print(f"{t:<8} | {acc_pct:>5.1f}%    | {st['passed']}/{st['total']:<5} | {avg_turns:<10} | {avg_time:<10}s")
    print("-" * 55)
    print(f"Overall Accuracy: {final_data['overall_accuracy_percent']}% ({final_data['passed_count']}/{final_data['total_runs']} passed)")
    print(f"Scorecard saved to: {summary_file}")
    print("=================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_100_benchmark(
        tasks=task_list,
        samples_per_task=args.count,
        max_workers=args.workers,
        model=args.model,
        max_turns=args.max_turns,
        resume=not args.no_resume
    )
