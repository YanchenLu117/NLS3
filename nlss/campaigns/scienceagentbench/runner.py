"""ScienceAgentBench runner: stratified subset x arms -> predicted programs.

Resolves each task's real dataset dir, injects absolute data paths, and generates
programs with self-debug validation (memory arms) or the official SAB agent
(coder). Programs are written under out/<run>/<arm>/<task>/pred_<gold>.

NOTE (output-path contract): the official SAB grader (run_eval.py/compute_scores)
runs each predicted program from the SAB repo root and then requires the program
to have written its DATA OUTPUT to <repo_root>/pred_results/<task.output_fname>.
The generated program file itself lives under our results tree (out/.../pred_<gold>).
These two paths are distinct: `pred_file` = where we save the generated source;
`task_out` = the absolute path the program must write its data result to, so the
official eval_script can find it.
"""
import json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
from nlss.campaigns.scienceagentbench.tasks import load_metadata, stratify
from nlss.campaigns.scienceagentbench.agent import (gen_program_with_self_debug,
                                                    inject_backbone, TunnelEngine,
                                                    propose_memory)

SAB = REPO / "external" / "repos" / "scienceagentbench"
BENCH = SAB / "benchmark"
sys.path.insert(0, str(SAB))


def build_datadir_index():
    """file_name -> dataset dir, from the extracted benchmark/datasets."""
    idx = {}
    dd = BENCH / "datasets"
    if dd.is_dir():
        for d in dd.iterdir():
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.is_file():
                        idx.setdefault(f.name, d)
    return idx


def resolve_datadir(task, idx):
    tree = task.get("dataset_folder_tree") or ""
    for line in tree.splitlines():
        for tok in line.replace("|--", " ").replace("|", " ").split():
            if tok in idx:
                return idx[tok]
    for tok in re.findall(r"[A-Za-z0-9_.+-]+", tree):
        if tok in idx:
            return idx[tok]
    return None


def data_schema(datadir, max_files=4):
    """Inspect the task's real data files (an observed signal) into a compact
    schema string: file names, tabular columns, and row counts. This is used to
    ground the memory arms' recovered state in the actual data."""
    if not datadir or not Path(datadir).is_dir():
        return "(no data dir)"
    import csv
    lines = []
    files = sorted(Path(datadir).iterdir())[:max_files]
    for f in files:
        if not f.is_file():
            continue
        try:
            if f.suffix.lower() in (".csv", ".tsv"):
                sep = "\t" if f.suffix.lower() == ".tsv" else ","
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    head = [fh.readline() for _ in range(3)]
                nrows = sum(1 for _ in fh) if False else 0
                cols = head[0].strip().split(sep) if head and head[0].strip() else []
                lines.append("%s: cols=%s, rows(first lines)=%d" % (
                    f.name, cols, len([h for h in head if h.strip()])))
            else:
                lines.append("%s (%d bytes)" % (f.name, f.stat().st_size))
        except Exception as e:
            lines.append("%s (read-error %s)" % (f.name, type(e).__name__))
    return "DATASET SCHEMA (observed from files):\n" + "\n".join(lines)


def run_arm(arm, task, idx, outdir):
    iid = str(task["instance_id"])
    td = outdir / arm / iid
    td.mkdir(parents=True, exist_ok=True)
    datadir = resolve_datadir(task, idx)
    data_files = sorted(f.name for f in (datadir.rglob("*") if datadir and datadir.is_dir() else []))
    # Where we save the generated PRED program on disk (under our results tree).
    pred_file = str(td / ("pred_" + task["gold_program_name"]))
    # Absolute path the program must write its DATA OUTPUT to, matching the
    # official grader's <repo_root>/pred_results/<output_fname> lookup.
    task_out = str(SAB / task["output_fname"])
    (SAB / "pred_results").mkdir(parents=True, exist_ok=True)

    if arm == "coder":
        from agent import ScienceAgent
        ag = inject_backbone(ScienceAgent("gpt-4o"), TunnelEngine())
        taskd = {"task_inst": task["task_inst"],
                 "dataset_path": str(datadir) if datadir else "",
                 "dataset_folder_tree": task["dataset_folder_tree"],
                 "dataset_preview": task["dataset_preview"],
                 "output_fname": task["output_fname"]}
        ag.solve_task(taskd, pred_file)
        return {"arm": arm, "task": iid, "pred": pred_file,
                "exists": Path(pred_file).exists(), "datadir": str(datadir), "kind": "sab"}

    dpath = str(datadir) if datadir else ""
    memory = propose_memory(arm, task, data_files, schema=data_schema(datadir))
    code = gen_program_with_self_debug(
        task["task_inst"], data_files, dpath, task_out, task.get("dataset_preview"),
        memory_state=memory)
    Path(pred_file).write_text(code)
    return {"arm": arm, "task": iid, "pred": pred_file,
            "exists": Path(pred_file).exists(), "datadir": dpath, "kind": "memory"}


def run(config, data_root=None, arms="nlss,hypoth,generic,coder", seed=0, want=16,
        out="results/scienceagentbench/pilot"):
    meta = load_metadata()
    sub = stratify(meta, seed=seed, want=want)
    idx = build_datadir_index()
    outdir = Path(out); outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in sub:
        for arm in arms.split(","):
            arm = arm.strip()
            if not arm:
                continue
            rec = run_arm(arm, t, idx, outdir)
            manifest.append(dict(out_dir=outdir.name, **rec))
            print(f"[{arm}] {t['instance_id']} exists={rec['exists']} datadir={rec.get('datadir')}", flush=True)
    (outdir / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("DONE tasks=", len(sub), flush=True)
