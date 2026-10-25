from typing import List, Dict, Any

PYTHON_REPL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_repl",
        "description": (
            "Executes Python code in a persistent interactive REPL session with access to the document. "
            "Variables and imports survive across turns. Output is captured and printed back. "
            "Pre-injected helper functions: grep(pattern, max_matches=30, context_lines=0), "
            "read_lines(start_line, end_line), and file_info()."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code snippet to run in the REPL."
                }
            },
            "required": ["code"]
        }
    }
}

ALL_TOOLS: List[Dict[str, Any]] = [PYTHON_REPL_TOOL]
