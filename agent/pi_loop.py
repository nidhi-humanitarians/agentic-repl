import re
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from agentic_search_pi.agent.client import ParticleAIClient
    from agentic_search_pi.agent.prompts import SYSTEM_SEARCH_PROMPT
    from agentic_search_pi.agent.tools_schema import ALL_TOOLS
    from agentic_search_pi.repl.session import PythonREPL
    from agentic_search_pi.config.settings import DEFAULT_MODEL
except ImportError:
    from agent.client import ParticleAIClient
    from agent.prompts import SYSTEM_SEARCH_PROMPT
    from agent.tools_schema import ALL_TOOLS
    from repl.session import PythonREPL
    from config.settings import DEFAULT_MODEL

class PiAgent:
    """
    Pi Agent execution harness for deep document search via code REPL.
    """
    def __init__(
        self,
        corpus_path: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = 15,
        max_output_chars: int = 3500
    ):
        self.corpus_path = str(Path(corpus_path).resolve())
        self.model = model
        self.max_turns = max_turns
        
        # Initialize client and REPL sandbox
        self.client = ParticleAIClient()
        self.repl = PythonREPL(default_corpus_path=self.corpus_path, max_output_chars=max_output_chars)
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_SEARCH_PROMPT}
        ]
        self.trajectory: List[Dict[str, Any]] = []

    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """Fallback to extract Python code block if model output code in markdown or DSML syntax."""
        # 1. Standard markdown code blocks
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # 2. DeepSeek DSML tool invocation syntax
        dsml_match = re.search(r'<[｜|]DSML[｜|]\s+parameter\s+name=["\']code["\']\s+string=["\'](.*?)["\']', text, re.DOTALL)
        if dsml_match:
            # unescape newlines if string-encoded
            code_str = dsml_match.group(1)
            code_str = code_str.replace("\\n", "\n").replace('\\"', '"')
            return code_str.strip()

        return None

    def search(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes the agentic search loop for a user query.
        """
        user_prompt = (
            f"Corpus file to search: {self.corpus_path}\n"
            f"Question: {query}\n\n"
            f"Please use the `python_repl` tool to investigate the document and find the factual answer."
        )
        self.messages.append({"role": "user", "content": user_prompt})
        
        start_time = time.time()
        final_answer = None
        turn = 0

        if verbose:
            print(f"\n==================== STARTING AGENTIC SEARCH ====================")
            print(f"Model: {self.model}")
            print(f"Corpus: {Path(self.corpus_path).name}")
            print(f"Query: {query}\n")

        while turn < self.max_turns:
            turn += 1
            if verbose:
                print(f"--- [Turn {turn}/{self.max_turns}] Invoking Model ---")

            response = self.client.chat_completion(
                messages=self.messages,
                model=self.model,
                tools=ALL_TOOLS,
                temperature=0.0
            )

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            tool_calls = getattr(msg, "tool_calls", None)

            if verbose and reasoning:
                print(f"[Model Reasoning]: {reasoning[:200]}..." if len(reasoning) > 200 else f"[Model Reasoning]: {reasoning}")
            if verbose and content:
                print(f"[Model Thought/Output]:\n{content}\n")

            # Case A: Model issued structured tool_calls
            if tool_calls:
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in tool_calls
                    ]
                }
                self.messages.append(assistant_msg)

                for tc in tool_calls:
                    fn_name = tc.function.name
                    args_raw = tc.function.arguments
                    try:
                        args = json.loads(args_raw)
                        code = args.get("code", "")
                    except Exception as e:
                        code = args_raw

                    if verbose:
                        print(f"[Executing Tool: {fn_name}]\n>>> Code:\n{code}")

                    # Execute in persistent REPL
                    repl_res = self.repl.execute(code)
                    output_str = repl_res["output"]

                    if verbose:
                        preview = output_str[:400] + ("..." if len(output_str) > 400 else "")
                        print(f"<<< REPL Output ({len(output_str)} chars):\n{preview}\n")

                    self.trajectory.append({
                        "turn": turn,
                        "type": "tool_call",
                        "code": code,
                        "output": output_str,
                        "success": repl_res["success"]
                    })

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": output_str
                    })
                continue

            # Case B: Model wrote code in markdown without structured tool call
            fallback_code = self._extract_code_from_text(content)
            if fallback_code and ("grep(" in fallback_code or "read_lines(" in fallback_code or "open(" in fallback_code):
                if verbose:
                    print(f"[Detected Markdown Code Block - Executing in REPL]\n>>> Code:\n{fallback_code}")

                repl_res = self.repl.execute(fallback_code)
                output_str = repl_res["output"]

                if verbose:
                    preview = output_str[:400] + ("..." if len(output_str) > 400 else "")
                    print(f"<<< REPL Output ({len(output_str)} chars):\n{preview}\n")

                self.trajectory.append({
                    "turn": turn,
                    "type": "markdown_code",
                    "code": fallback_code,
                    "output": output_str,
                    "success": repl_res["success"]
                })

                self.messages.append({"role": "assistant", "content": content})
                self.messages.append({
                    "role": "user",
                    "content": f"[Observation from REPL execution]:\n{output_str}"
                })
                continue

            # Case C: No tool calls and no code blocks
            # If the model returned an empty string, prompt it to continue rather than exiting prematurely
            if not content.strip():
                if verbose:
                    print("[Model returned empty response - prompting to continue execution...]")
                self.messages.append({"role": "assistant", "content": content or ""})
                self.messages.append({
                    "role": "user",
                    "content": "Please proceed with your next action: execute your next step using `python_repl` or formulate your final answer."
                })
                continue

            final_answer = content.strip()
            self.trajectory.append({
                "turn": turn,
                "type": "final_answer",
                "content": final_answer
            })
            if verbose:
                print(f"==================== SEARCH COMPLETE ====================")
                print(f"Final Answer:\n{final_answer}\n")
            break

        # Graceful Wrap-Up: If budget reached while gathering evidence, prompt for final synthesis
        if final_answer is None:
            if verbose:
                print("\n[Budget Reached] Requesting final answer synthesis from gathered evidence...")
            self.messages.append({
                "role": "user",
                "content": (
                    f"You have finished searching. Based on all the observations, lines, and facts discovered "
                    f"in the REPL executions above, please provide your final factual answer to the question: '{query}'. "
                    f"State your answer in plain text without code or tool calls."
                )
            })
            resp = self.client.chat_completion(
                messages=self.messages,
                model=self.model,
                temperature=0.0,
                max_tokens=512
            )
            raw_final = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, "reasoning_content", "") or ""

            # Protocol auto-healing: if the model emitted raw tool tokens, auto-heal and execute
            if "<｜DSML｜" in raw_final or "<tool_call>" in raw_final or "```python" in raw_final:
                extracted = self._extract_code_from_text(raw_final)
                if extracted and ("grep(" in extracted or "read_lines(" in extracted or "open(" in extracted):
                    repl_res = self.repl.execute(extracted)
                    out_s = repl_res["output"]
                    followup = self.client.chat_completion(
                        messages=[
                            *self.messages,
                            {"role": "assistant", "content": raw_final},
                            {"role": "user", "content": f"[Final Tool Observation]: {out_s}\nNow state your definitive answer directly in one sentence without tools."}
                        ],
                        model=self.model,
                        temperature=0.0,
                        max_tokens=256
                    )
                    raw_final = followup.choices[0].message.content or raw_final

            # Strip any residual protocol tags
            clean_final = re.sub(r"<[｜|]DSML[｜|][^>]*>", "", raw_final)
            clean_final = re.sub(r"</?[a-zA-Z0-9_\-]+>", "", clean_final).strip()
            final_answer = clean_final if clean_final else (reasoning.strip() if reasoning else raw_final)

            self.trajectory.append({
                "turn": turn + 1,
                "type": "final_synthesis",
                "content": final_answer
            })
            if verbose:
                print(f"==================== SEARCH COMPLETE (SYNTHESIS) ====================")
                print(f"Final Answer:\n{final_answer}\n")

        total_time = round(time.time() - start_time, 2)
        return {
            "query": query,
            "final_answer": final_answer,
            "total_turns": turn,
            "total_time_seconds": total_time,
            "repl_executions": self.repl.execution_count,
            "trajectory": self.trajectory
        }
