import re
from typing import Dict, Any

def check_answer_match(ground_truth: str, final_answer: str) -> Dict[str, Any]:
    """
    Evaluates whether the agent's final answer satisfies the ground-truth target.
    Handles semantic equivalence for counts (0 / zero / none / nothing),
    binary affirmations (yes / no), and entity word boundaries.
    """
    if not final_answer or not final_answer.strip():
        return {
            "passed": False,
            "extracted": "",
            "reason": "Final answer is empty or missing"
        }

    gt = ground_truth.strip().lower()
    raw = final_answer.strip()
    ans_lower = raw.lower()

    # Extract leading line or "Answer:" clause if available
    answer_clause = ""
    ans_match = re.search(r"(?:answer|result|conclusion)\s*[:—\-]\s*(.+?)(?:\n|$)", ans_lower)
    if ans_match:
        answer_clause = ans_match.group(1).strip()
    else:
        # First non-empty line
        lines = [l.strip() for l in ans_lower.splitlines() if l.strip()]
        answer_clause = lines[0] if lines else ans_lower

    # 1. Binary Tasks (QA6, QA9, QA10)
    if gt == "yes":
        # If the response explicitly starts with an affirmative declaration
        if re.search(r"^(?:[\*\#\s_]*)(?:yes|true)\b", ans_lower) or re.search(r"(?:answer|conclusion)\s*[:—\-]\s*(?:[\*\#\s_]*)(?:yes|true)\b", ans_lower):
            return {"passed": True, "extracted": "yes", "reason": "Leading affirmative confirmation"}
        is_yes_lead = bool(re.search(r"\b(yes|true|confirmed)\b", answer_clause))
        is_no_lead = bool(re.search(r"\b(no|not|neither|never)\b", answer_clause))
        if is_yes_lead and not is_no_lead:
            return {"passed": True, "extracted": "yes", "reason": "Affirmative confirmation"}
        if bool(re.search(r"\b(yes|is in the)\b", ans_lower)) and not bool(re.search(r"\b(no|not in the)\b", answer_clause)):
            return {"passed": True, "extracted": "yes", "reason": "Affirmative mention"}
        return {"passed": False, "extracted": "no/unclear", "reason": "Expected 'yes'"}

    elif gt == "no":
        # If the response explicitly starts with a negative declaration
        if re.search(r"^(?:[\*\#\s_]*)(?:no|false)\b", ans_lower) or re.search(r"(?:answer|conclusion)\s*[:—\-]\s*(?:[\*\#\s_]*)(?:no|false)\b", ans_lower):
            return {"passed": True, "extracted": "no", "reason": "Leading negative confirmation"}
        is_no_lead = bool(re.search(r"\b(no|not|false|never)\b", answer_clause))
        is_yes_lead = bool(re.search(r"\b(yes|true)\b", answer_clause))
        if is_no_lead and not is_yes_lead:
            return {"passed": True, "extracted": "no", "reason": "Negative confirmation"}
        if bool(re.search(r"\b(no|not in the|neither)\b", ans_lower)) and not bool(re.search(r"^\s*yes\b", answer_clause)):
            return {"passed": True, "extracted": "no", "reason": "Negative statement"}
        return {"passed": False, "extracted": "yes/unclear", "reason": "Expected 'no'"}

    # 2. Count & Quantity Tasks (QA7)
    elif gt in ["none", "nothing", "0", "zero"]:
        # Canonical zero: 0, zero, none, nothing, empty-handed, no objects
        zero_patterns = r"\b(none|nothing|zero|0|empty-handed|no objects|not carrying any|0 objects)\b"
        if re.search(zero_patterns, answer_clause) or re.search(zero_patterns, ans_lower):
            # Check it's not asserting positive possession like "is carrying 1 object"
            if not re.search(r"\b(1 object|one object|2 objects|two objects)\b", ans_lower):
                return {"passed": True, "extracted": "none/0", "reason": "Canonical zero"}
        return {"passed": False, "extracted": "positive quantity", "reason": "Expected 'none/nothing'"}

    elif gt in ["one", "1"]:
        if re.search(r"\b(one|1|single)\b", answer_clause) or re.search(r"\b(one|1)\s+object\b", ans_lower):
            if not re.search(r"\b(0|zero|none|nothing|two|2)\b", answer_clause):
                return {"passed": True, "extracted": "one/1", "reason": "Canonical one"}
        return {"passed": False, "extracted": "other count", "reason": "Expected 'one'"}

    elif gt in ["two", "2"]:
        if re.search(r"\b(two|2)\b", answer_clause) or re.search(r"\b(two|2)\s+objects\b", ans_lower):
            return {"passed": True, "extracted": "two/2", "reason": "Canonical two"}
        return {"passed": False, "extracted": "other count", "reason": "Expected 'two'"}

    elif gt in ["three", "3"]:
        if re.search(r"\b(three|3)\b", answer_clause) or re.search(r"\b(three|3)\s+objects\b", ans_lower):
            return {"passed": True, "extracted": "three/3", "reason": "Canonical three"}
        return {"passed": False, "extracted": "other count", "reason": "Expected 'three'"}

    # 3. Named Entities, Locations, Objects (QA1, QA2, QA3, QA4, QA5, QA8)
    else:
        # Strict word boundary match
        pattern = rf"\b{re.escape(gt)}\b"
        if re.search(pattern, ans_lower):
            return {"passed": True, "extracted": gt, "reason": f"Exact word boundary match for '{gt}'"}
        return {"passed": False, "extracted": "unmatched", "reason": f"Target '{gt}' not found"}

if __name__ == "__main__":
    # Self-test unit assertions
    assert check_answer_match("kitchen", "**The football is in the kitchen.**")["passed"]
    assert check_answer_match("none", "## Answer: **0 (zero)** — Mary is not carrying any objects.")["passed"]
    assert check_answer_match("nothing", "**Answer: Nothing — Daniel is empty-handed.**")["passed"]
    assert check_answer_match("one", "**Answer: 1 object — the milk.**")["passed"]
    assert check_answer_match("yes", "**Yes — Julie is in the cinema.**")["passed"]
    assert check_answer_match("no", "## Answer: **No, Sandra is not in the bedroom.**")["passed"]
    assert not check_answer_match("yes", "No, Sandra is not in the bedroom.")["passed"]
    assert not check_answer_match("kitchen", "The football is in the garden.")["passed"]
    print("All evaluator unit tests passed successfully!")
