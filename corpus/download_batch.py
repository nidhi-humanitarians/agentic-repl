import json
import argparse
import time
import requests
from pathlib import Path

def stream_and_extract_samples(
    task: str,
    length: str = "1M",
    target_count: int = 10,
    output_dir: str = "corpus"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Check which sample indices already exist on disk
    needed_indices = []
    for idx in range(target_count):
        # Index 0 can be either babilong_qaX_1M.txt or babilong_qaX_1M_s0.txt
        f1 = out_path / f"babilong_{task}_{length}_s{idx}.txt"
        f0 = out_path / f"babilong_{task}_{length}.txt" if idx == 0 else f1
        if not f1.exists() and not f0.exists():
            needed_indices.append(idx)

    if not needed_indices:
        print(f"[{task.upper()}] All {target_count} samples (s0-s{target_count-1}) already exist. Skipping download.")
        return

    print(f"[{task.upper()}] Fetching samples {needed_indices} from Hugging Face ({length})...")
    url = f"https://huggingface.co/datasets/RMT-Team/babilong/resolve/main/data/{task}/{length}.json"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()

    buffer = ""
    current_idx = 0
    max_needed = max(needed_indices)
    extracted_count = 0

    # Stream in chunks
    for chunk in response.iter_content(chunk_size=1024 * 512):
        buffer += chunk.decode("utf-8", errors="ignore")
        
        while True:
            start_pos = buffer.find("{")
            if start_pos == -1:
                break
                
            brace_count = 0
            in_string = False
            escape = False
            end_pos = -1
            
            for i in range(start_pos, len(buffer)):
                c = buffer[i]
                if c == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                escape = (c == "\\" and not escape)
                
            if end_pos != -1:
                if current_idx in needed_indices:
                    sample_json_str = buffer[start_pos : end_pos + 1]
                    try:
                        sample = json.loads(sample_json_str)
                        q = sample.get("question", "")
                        tgt = sample.get("target", "")
                        text = sample.get("input", "")

                        # Save text & meta
                        txt_path = out_path / f"babilong_{task}_{length}_s{current_idx}.txt"
                        meta_path = out_path / f"babilong_{task}_{length}_s{current_idx}_meta.json"
                        
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text)

                        lines = len(text.splitlines())
                        meta_data = {
                            "benchmark": "BABILong",
                            "task": task,
                            "length_tier": length,
                            "sample_index": current_idx,
                            "corpus_file": str(txt_path),
                            "question": q,
                            "target": tgt,
                            "total_lines": lines,
                            "total_chars": len(text),
                            "approx_pages": round(len(text) / 1800, 1)
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta_data, f, indent=2)

                        extracted_count += 1
                        print(f"  -> Saved s{current_idx}: {lines} lines, ~{meta_data['approx_pages']} pages (Q: '{q}' | Target: '{tgt}')")
                    except Exception as e:
                        print(f"  Error parsing sample {current_idx}: {e}")

                current_idx += 1
                buffer = buffer[end_pos + 1 :]
                
                if current_idx > max_needed:
                    break
            else:
                break
                
        if current_idx > max_needed:
            break

    response.close()
    print(f"[{task.upper()}] Done! Extracted {extracted_count} new samples.\n")

def download_all_qa_batch(tasks=("qa1","qa2","qa3","qa4","qa5","qa6","qa7","qa8","qa9","qa10"), target_count=10):
    repo_dir = Path(__file__).resolve().parent.parent
    corpus_dir = repo_dir / "corpus"
    
    t0 = time.time()
    for task in tasks:
        stream_and_extract_samples(task=task, length="1M", target_count=target_count, output_dir=str(corpus_dir))
    print(f"All downloads finished in {round(time.time() - t0, 1)} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    download_all_qa_batch(tasks=task_list, target_count=args.count)
