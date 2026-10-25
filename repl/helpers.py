import re
from pathlib import Path
from typing import List, Dict, Any, Optional

class SearchHelpers:
    """
    Search and slicing utilities injected directly into the REPL environment.
    Allows the agent to quickly probe massive files without writing boilerplate.
    """
    def __init__(self, default_file_path: Optional[str] = None):
        self.default_file_path = default_file_path

    def _resolve_path(self, file_path: Optional[str]) -> str:
        target = file_path or self.default_file_path
        if not target:
            raise ValueError("No file path specified and no default corpus file set.")
        return str(target)

    def grep(
        self,
        pattern: str,
        file_path: Optional[str] = None,
        max_matches: int = 30,
        case_sensitive: bool = False,
        context_lines: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Searches for regex/string pattern in file.
        Returns list of matches with line numbers and text.
        """
        target = self._resolve_path(file_path)
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        matches = []
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(total_lines, i + context_lines + 1)
                context = [lines[idx].rstrip("\r\n") for idx in range(start, end)]
                
                matches.append({
                    "line_number": i + 1,
                    "matched_line": line.strip(),
                    "context": context if context_lines > 0 else None
                })
                if len(matches) >= max_matches:
                    break
        return matches

    def read_lines(
        self,
        start_line: int,
        end_line: int,
        file_path: Optional[str] = None
    ) -> str:
        """
        Reads a 1-indexed bounded range of lines from the file.
        """
        target = self._resolve_path(file_path)
        start_idx = max(0, start_line - 1)
        end_idx = end_line
        
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        selected = lines[start_idx:end_idx]
        formatted = [f"{start_idx + idx + 1}: {line.rstrip()}" for idx, line in enumerate(selected)]
        return "\n".join(formatted)

    def file_info(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns line count, character count, and file size in KB.
        """
        target = self._resolve_path(file_path)
        p = Path(target)
        size_kb = p.stat().st_size / 1024
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {
            "file": p.name,
            "total_lines": len(lines),
            "size_kb": round(size_kb, 2)
        }
