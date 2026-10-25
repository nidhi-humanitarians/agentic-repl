import json
import argparse
import requests
from pathlib import Path

def download_babilong_sample(
    task: str = "qa1",
    length: str = "1M",
    sample_idx: int = 1,
    output_dir: str = "corpus"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    url = f"https://huggingface.co/datasets/RMT-Team/babilong/resolve/main/data/{task}/{length}.json"
    print(f"Connecting to Hugging Face for BABILong task '{task}' at length '{length}' (Sample #{sample_idx})...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()

    buffer = ""
    sample = None
    current_idx = 0
    
    # Stream through JSON array and extract target sample_idx
    for chunk in response.iter_content(chunk_size=1024 * 256):
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
                if current_idx == sample_idx:
                    sample_json_str = buffer[start_pos : end_pos + 1]
                    sample = json.loads(sample_json_str)
                    break
                else:
                    current_idx += 1
                    buffer = buffer[end_pos + 1 :]
            else:
                # Need more data in buffer
                break
                
        if sample is not None:
            break

    if not sample:
        raise RuntimeError(f"Failed to extract sample #{sample_idx} for {task} ({length}).")

    question = sample.get("question", "")
    target = sample.get("target", "")
    text_content = sample.get("input", "")

    # Save corpus text file with sample_idx in filename
    suffix = f"_s{sample_idx}" if sample_idx > 0 else ""
    txt_filename = f"babilong_{task}_{length}{suffix}.txt"
    txt_path = out_path / txt_filename
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    lines_count = len(text_content.splitlines())
    char_count = len(text_content)

    meta_filename = f"babilong_{task}_{length}{suffix}_meta.json"
    meta_path = out_path / meta_filename
    meta_data = {
        "benchmark": "BABILong",
        "task": task,
        "length_tier": length,
        "sample_index": sample_idx,
        "corpus_file": str(txt_path),
        "question": question,
        "target": target,
        "total_lines": lines_count,
        "total_chars": char_count,
        "approx_pages": round(char_count / 1800, 1)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print(f"Saved: {txt_path.name} ({lines_count} lines, ~{meta_data['approx_pages']} pages)")
    print(f"Question: {question} | Target: {target}")

    return meta_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qa1")
    parser.add_argument("--length", default="1M")
    parser.add_argument("--sample_idx", type=int, default=1)
    parser.add_argument("--output_dir", default="corpus")
    args = parser.parse_args()
    
    download_babilong_sample(task=args.task, length=args.length, sample_idx=args.sample_idx, output_dir=args.output_dir)
