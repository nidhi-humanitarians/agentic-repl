import sys
import io
import ast
import traceback
from typing import Dict, Any, Optional
from agentic_search_pi.repl.helpers import SearchHelpers

class PythonREPL:
    """
    A persistent, interactive Python REPL session.
    State (variables, functions, imports) is retained across executions.
    Outputs are bounded to protect LLM context windows.
    """
    def __init__(self, default_corpus_path: Optional[str] = None, max_output_chars: int = 3500):
        self.max_output_chars = max_output_chars
        self.default_corpus_path = default_corpus_path
        self.execution_count = 0
        
        # Initialize persistent execution environment
        self.helpers = SearchHelpers(default_file_path=default_corpus_path)
        self.env: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "helpers": self.helpers,
            "grep": self.helpers.grep,
            "read_lines": self.helpers.read_lines,
            "file_info": self.helpers.file_info,
            "default_corpus_path": default_corpus_path,
        }
        
        # Pre-import standard libraries into REPL
        exec("import re, json, math, os, sys, collections", self.env)

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Executes a Python code block within the persistent session.
        Returns a dict containing stdout/stderr, success status, and error details.
        """
        self.execution_count += 1
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        success = True
        error_msg = None
        result_value = None

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Parse AST to check if last statement is an expression
            parsed = ast.parse(code)
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                # Separate statements from the final expression
                *body_stmts, last_expr = parsed.body
                
                # Execute preceding statements
                if body_stmts:
                    mod = ast.Module(body=body_stmts, type_ignores=[])
                    compiled_stmts = compile(mod, filename="<repl>", mode="exec")
                    exec(compiled_stmts, self.env)
                
                # Evaluate final expression
                expr_ast = ast.Expression(body=last_expr.value)
                compiled_expr = compile(expr_ast, filename="<repl>", mode="eval")
                result_value = eval(compiled_expr, self.env)
                
                # If result is not None, format and print to captured stdout
                if result_value is not None:
                    if isinstance(result_value, (dict, list)):
                        import json
                        try:
                            print(json.dumps(result_value, indent=2))
                        except Exception:
                            print(repr(result_value))
                    else:
                        print(str(result_value))
            else:
                # Execute full block as standard statements
                compiled = compile(code, filename="<repl>", mode="exec")
                exec(compiled, self.env)

        except Exception:
            success = False
            error_msg = traceback.format_exc()
            stderr_capture.write(error_msg)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        raw_output = stdout_capture.getvalue()
        raw_error = stderr_capture.getvalue()
        
        combined_output = ""
        if raw_output:
            combined_output += raw_output
        if raw_error:
            if combined_output:
                combined_output += "\n--- STDERR / TRACEBACK ---\n"
            combined_output += raw_error

        if not combined_output.strip():
            combined_output = "[Executed successfully with no output]"

        # Enforce output length limit
        truncated = False
        if len(combined_output) > self.max_output_chars:
            total_len = len(combined_output)
            combined_output = (
                combined_output[:self.max_output_chars]
                + f"\n\n... [Output truncated: showing first {self.max_output_chars} of {total_len} chars. Refine your query or use slicing.]"
            )
            truncated = True

        return {
            "output": combined_output.strip(),
            "success": success,
            "truncated": truncated,
            "execution_count": self.execution_count
        }

    def reset(self):
        """Resets the environment."""
        self.__init__(default_corpus_path=self.default_corpus_path, max_output_chars=self.max_output_chars)
