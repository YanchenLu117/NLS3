"""ScienceAgentBench agent compatibility layer + robust program generation.

Two layers:
1. TunnelEngine: inject our backbone into SAB's ScienceAgent (no litellm patch),
   so the official CodeAct/self-debug loop runs on the tunnel model.
2. Memory arms (nlss/hypoth/generic): robust self-debugging code generation —
   syntax is validated and regenerated on failure (so a weak backbone still
   emits valid programs), and the resolved absolute data path is injected so the
   program loads the real dataset. Each arm conditions generation on a DIFFERENT
   memory/state (see propose_memory) so the arms are genuinely compared.
"""
import os, re

MODEL = os.environ.get("DB_MODEL", "deepseek-v4-flash")
BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18002/v1")
KEY = os.environ.get("OPENAI_API_KEY", "zzlzl")


class TunnelEngine:
    def __init__(self, model=MODEL):
        self.model = model
        self.llm_engine_name = model  # SAB get_sys_msg reads this

    def respond(self, user_input, temperature=0.2, top_p=0.95):
        from openai import OpenAI
        c = OpenAI(base_url=BASE, api_key=KEY)
        msgs = ([{"role": "user", "content": user_input}]
                if not isinstance(user_input, (list, tuple)) else user_input)
        r = c.chat.completions.create(
            model=self.model, messages=msgs, temperature=temperature, top_p=top_p)
        txt = r.choices[0].message.content or ""
        u = getattr(r, "usage", None)
        pin = getattr(u, "prompt_tokens", 0) if u else 0
        pout = getattr(u, "completion_tokens", 0) if u else 0
        return txt, pin, pout


ZERO_COST = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
             "max_tokens": 8192}


def inject_backbone(agent, llm_engine=None):
    agent.llm_engine = llm_engine or TunnelEngine()
    agent.llm_cost = dict(ZERO_COST)
    return agent


def _llm(prompt, temperature=0.3, max_tokens=2400):
    from openai import OpenAI
    c = OpenAI(base_url=BASE, api_key=KEY)
    r = c.chat.completions.create(model=MODEL,
                                  messages=[{"role": "user", "content": prompt}],
                                  temperature=temperature, max_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    if text.lstrip().startswith(("import ", "from ", "def ", "#", "print(")):
        return text
    return text


def _syntax_ok(code):
    try:
        compile(code, "<gen>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} (line {e.lineno})"


def propose_memory(arm, task, data_files, schema=""):
    """Arm-specific memory/state conditioned into generation (v2: schema-grounded).

    generic: no extra recovered state (fixed plan) - weak baseline.
    hypoth:  LLM proposes N DISTINCT candidate hypotheses/approach families,
             grounded on the observed schema.
    nlss:    LLM performs explicit solution-space recovery - enumerates decisional
             dimensions (preprocessing / feature+model / metric / output-contract /
             robustness) with candidate options (the solution space), then SELECTS one
             high-recovery configuration per dimension and states the output contract.
             All arms get the SAME observed schema; only the memory structure differs.
    """
    base = "Task:\n%s\n\nData files: %s\n%s" % (
        (task.get("task_inst") or "")[:2500],
        [os.path.basename(f) for f in data_files[:20]],
        schema)
    if arm == "generic":
        return ("Generic solving plan:\n"
                "1. Load each dataset with absolute paths and inspect schema.\n"
                "2. Pin down the exact requested output and its format.\n"
                "3. Implement the analysis/model, guarding missing columns.\n"
                "4. Save the required output to the given absolute output path.")
    if arm == "hypoth":
        prompt = (base + "\n\nUsing ONLY the observed schema above, propose 3-5 DISTINCT "
                  "hypotheses / high-level approach families for solving this task (different "
                  "method families). For each, name the exact columns/inputs it would use from "
                  "the schema. Output ONLY a compact bullet list H1..HN, one line each. No code.")
        return "[hypoth arm - candidate hypotheses]\n" + (_llm(prompt, temperature=0.6, max_tokens=1000) or "none")
    if arm == "nlss":
        prompt = (base + "\n\nPerform explicit SOLUTION-SPACE RECOVERY grounded on the observed "
                  "schema above. Step 1: enumerate the key decisional dimensions of this task "
                  "(feature/column selection, preprocessing, model/analysis family, metric/objective, "
                  "OUTPUT CONTRACT, failure modes) each with 2-4 candidate options that actually "
                  "exist in the schema - this is the solution space. Step 2: for each dimension "
                  "SELECT the single option with highest recovery probability (most likely to yield "
                  "a correct, runnable, well-grounded program that reads the real columns and "
                  "writes the required output). The OUTPUT CONTRACT dimension MUST specify the "
                  "exact output file, columns/format, and how to derive them from the schema. "
                  "Output ONLY the compact structured 'recovered state': one line per dimension "
                  "as 'dim -> selected option (rationale)'. No code.")
        return "[nlss arm - recovered solution-space state]\n" + (_llm(prompt, temperature=0.5, max_tokens=1200) or "none")
    return ""


def gen_program_with_self_debug(task_inst, data_files, data_path, output_fname,
                                preview, memory_state="", n_retry=3):
    """Self-debugging code generation: syntax-check + regenerate until valid."""
    data_list = "\n".join(f"- {os.path.basename(f)}" for f in data_files[:20])
    prompt = (
        "You are a data-driven scientific programming agent. Write ONE complete, "
        "self-contained Python program that solves the task below and RUNS WITHOUT ERROR "
        "against the provided data files.\n\n"
        f"TASK:\n{task_inst}\n\n"
        f"DATA FILES (in directory {data_path}):\n{data_list}\n\n"
        "Load each dataset with FULL ABSOLUTE paths via os.path.join(<THE DATA DIRECTORY>, "
        f'filename) where <THE DATA DIRECTORY> = r"{data_path}" — never rely on CWD.\n\n'
        f"OUTPUT FILE (absolute): {output_fname}\n\n"
        + (f"RECOVERED STATE:\n{memory_state}\n\n" if memory_state else "")
        + (f"DATA PREVIEW:\n{preview[:1500]}\n\n" if preview else "")
        + "Write only the final Python program (no prose). It must import nothing beyond "
        "standard + common data-science libraries, guard against missing columns, and "
        "save its output to the OUTPUT FILE."
    )
    code = _extract_code(_llm(prompt))
    for i in range(n_retry):
        ok, err = _syntax_ok(code)
        if ok:
            return code
        fix = (
            f"The last program had a SyntaxError:\n{err}\n\n"
            "Fix it and output ONLY the corrected, complete Python program (inside ```python ... ```):\n"
            f"{code[:1800]}"
        )
        code = _extract_code(_llm(fix))
    return code  # best effort
