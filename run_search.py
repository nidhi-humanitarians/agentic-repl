import json
import argparse
from pathlib import Path
from agentic_search_pi.agent.pi_loop import PiAgent
from agentic_search_pi.config.settings import DEFAULT_MODEL, CORPUS_DIR, EVALS_DIR

def run_main():
    parser = argparse.ArgumentParser(description="Run Pi Agent Deep Search via Code REPL")
    parser.add_argument(
        "--corpus",
        default=str(CORPUS_DIR / "babilong_qa2_128k.txt"),
        help="Path to corpus text file to search"
    )
    parser.add_argument(
        "--meta",
        default=str(CORPUS_DIR / "babilong_qa2_128k_meta.json"),
        help="Optional path to metadata file containing ground truth question and target"
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Specific question to search for. If omitted, loaded from metadata file."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Particle AI model to use (e.g. deepseek-v4.1-flash or glm-5.3-flash)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=12,
        help="Maximum turns allowed for the search loop"
    )
    args = parser.parse_args()

    # Determine query and ground truth
    ground_truth = None
    query = args.query

    meta_path = Path(args.meta)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if not query:
                query = meta.get("question")
            ground_truth = meta.get("target")

    if not query:
        query = "Where is the football?"

    print("\n========================================================")
    print("        AGENTIC SEARCH VIA PYTHON REPL (PI AGENT)       ")
    print("========================================================")
    print(f"Corpus Path:   {args.corpus}")
    print(f"Query:         {query}")
    if ground_truth:
        print(f"Ground Truth:  {ground_truth}")
    print(f"Model:         {args.model}")
    print("========================================================\n")

    agent = PiAgent(
        corpus_path=args.corpus,
        model=args.model,
        max_turns=args.max_turns
    )

    result = agent.search(query=query, verbose=True)

    # Evaluate against ground truth if available
    eval_result = "N/A"
    if ground_truth:
        answer_str = (result.get("final_answer") or "").lower()
        target_str = ground_truth.strip().lower()
        if target_str in answer_str:
            eval_result = "PASS (Ground truth matched!)"
        else:
            eval_result = "FAIL (Target not found in answer)"

    print("\n========================================================")
    print("                    EVALUATION SUMMARY                  ")
    print("========================================================")
    print(f"Final Answer:       {result.get('final_answer')}")
    if ground_truth:
        print(f"Expected Target:    {ground_truth}")
        print(f"Verification:       {eval_result}")
    print(f"Total Turns:        {result.get('total_turns')}")
    print(f"REPL Executions:    {result.get('repl_executions')}")
    print(f"Elapsed Time:       {result.get('total_time_seconds')}s")
    print("========================================================\n")

    # Save run record to evals directory
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out_record = EVALS_DIR / "last_run.json"
    result["evaluation"] = eval_result
    result["ground_truth"] = ground_truth
    with open(out_record, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Full trajectory record saved to: {out_record}")

if __name__ == "__main__":
    run_main()
