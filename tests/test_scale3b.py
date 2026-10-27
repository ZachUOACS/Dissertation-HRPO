"""3B scale check plumbing (submit_d2.sh scale3b, run_variant.slurm AUTO_EVAL_BASELINE,
eval_d2.py --baselines/--sizes, eval_quant.py default size, analyse_scale.py).

What must hold:
  * the tier's arguments name the run as a cold Dr. GRPO baseline (no "new-" prefix), and the id
    the job will score (baseline_id) is exactly that directory's id;
  * eval_d2 --baselines --sizes 3b --exact selects the 3B run only, never the 1.5B run that has
    the same id, and without --baselines greedy mode scores no baseline;
  * the auto-evaluation runs only after a successful training and does latent-off only for v3;
  * eval_quant defaults to 1.5B, so a 3B run never joins a 1.5B quantisation job;
  * analyse_scale pairs v1 and v3 per (size, task, seed) with the right McNemar counts.

    python -m pytest tests/test_scale3b.py -q
"""
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUBMIT = open(os.path.join(ROOT, "cluster/submit_d2.sh"), encoding="utf-8").read()
TIER = SUBMIT[SUBMIT.index("    scale3b)"):SUBMIT.index("    *)\n", SUBMIT.index("    scale3b)"))]


def tier_args():
    def value(name):
        parts = re.findall(rf'^\s*{name}="([^"]*)"', TIER, re.M)
        out = ""
        for p in parts:
            out = p.replace(f"${name}", out).strip()
        return out.split()
    return {"gsm8k": value("GSM8K_ARGS"), "math": value("MATH_ARGS")}


def bash_baseline_id(*argv):
    fn = re.search(r"^baseline_id\(\) \{.*?^\}", SUBMIT, re.M | re.S).group(0)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    out = subprocess.run([bash, "-c", fn + '\nbaseline_id "$@"', "x", *argv],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.parametrize("vnum,variant", [("1", "v1_grpo_baseline"), ("3", "v3_learned_lambda")])
@pytest.mark.parametrize("seed", ["42", "43"])
@pytest.mark.parametrize("task", ["gsm8k", "math"])
def test_tier_names_a_baseline_and_scores_that_id(vnum, variant, seed, task):
    import variant_common as vc
    args = vc.build_arg_parser(variant).parse_args(tier_args()[task] + ["--seed", seed])
    assert args.dr_grpo == "both" and args.repetition_penalty == 1.0
    name = os.path.basename(vc.experiment_dir("3b", variant, args))
    assert not name.startswith("new-")
    assert vc.parse_experiment_name(name) is not None
    assert name == "3b-" + bash_baseline_id(vnum, seed, task)
    if task == "math":
        assert args.max_completion_length == 1024 and args.per_device_train_batch_size == 8 \
            and args.gradient_accumulation_steps == 4 and args.dataset == "math"


def test_tier_matches_the_trained_1_5b_baselines():
    """Wherever the 1.5B baselines' train_summary.json exists (the cluster), every training
    argument the 3B tier passes equals what the 1.5B baseline of the same variant, task and seed
    trained with, except the model size. Keys added after a baseline was trained must be at the
    value it implicitly trained with (_FINGERPRINT_DEFAULTS, or off/None)."""
    import variant_common as vc
    ignore = {"model_size", "report_to", "math_root"}
    # Options added after the baselines trained, at the value every real run implicitly used:
    # --max_steps (smoke tests only, 23 October) is -1, one full epoch.
    IMPLICIT = {"max_steps": -1}
    checked = 0
    for variant in ("v1_grpo_baseline", "v3_learned_lambda"):
        for task in ("gsm8k", "math"):
            for seed in ("42", "43", "44"):
                args = vc.build_arg_parser(variant).parse_args(tier_args()[task] + ["--seed", seed])
                path = os.path.join(ROOT, vc.experiment_dir("1.5b", variant, args), "train_summary.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    trained = json.load(f)["args"]
                new = vars(args)
                for k in sorted(set(new) - ignore):
                    if k in trained:
                        assert trained[k] == new[k], (path, k, trained[k], new[k])
                    else:
                        assert new[k] in (vc._FINGERPRINT_DEFAULTS.get(k), IMPLICIT.get(k), None, False,
                                          "off"), (path, k)
                checked += 1
    if not checked:
        pytest.skip("no 1.5B baseline train_summary.json here (run on the cluster)")


def test_tier_settings():
    assert 'submit "${task}-v${v}-s${seed}" 3b "$v"' in TIER
    assert 'MATH_WALLTIME:-96:00:00' in TIER and 'GSM8K_WALLTIME:-48:00:00' in TIER
    assert "${JOB_MEM:+--mem=$JOB_MEM}" in SUBMIT


def test_run_variant_auto_eval_block():
    src = open(os.path.join(ROOT, "cluster/run_variant.slurm"), encoding="utf-8").read()
    block = src[src.index('if [[ -n "${AUTO_EVAL_BASELINE:-}"'):]
    assert "$train_status -eq 0" in block.splitlines()[0]
    assert block.count('--baselines --sizes "$MODEL_SIZE" --exact --variants "$AUTO_EVAL_BASELINE"') == 2
    off = block[block.index('if [[ "$VARIANT_NUM" != 1 ]]'):]
    assert "--latent_off" in off


def test_eval_quant_defaults_to_1_5b():
    src = open(os.path.join(ROOT, "eval_quant.py"), encoding="utf-8").read()
    assert 'parser.add_argument("--sizes", nargs="+", default=["1.5b"])' in src


def run(size, vid, base="v3_learned_lambda"):
    return {"model_size": size, "variant_id": vid, "base_variant": base, "checkpoint": f"{size}/{vid}",
            "gate_mode": "unit", "temperature": 0.5, "task": "math", "training_complete": True}


@pytest.fixture
def eval_d2(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
    stub = types.ModuleType("eval_all")
    stub.TASK_TEST_SIZE = {"gsm8k": 1319, "math": 500}
    for n in ("discover_experiments", "final_checkpoint", "get_run_temperature", "run_missing_evals"):
        setattr(stub, n, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "eval_all", stub)
    sys.modules.pop("eval_d2", None)
    mod = importlib.import_module("eval_d2")
    seen = {}
    monkeypatch.setattr(mod, "discover_new_runs", lambda: [
        run("1.5b", "v3_learned_lambda-rp1-drboth-dsmath-sl0.5")])
    monkeypatch.setattr(mod, "discover_baselines", lambda: [
        run("1.5b", "v3_learned_lambda-rp1-drboth-dsmath"),
        run("3b", "v3_learned_lambda-rp1-drboth-dsmath"),
        run("3b", "v3_learned_lambda-rp1-drboth")])
    monkeypatch.setattr(mod, "run_missing_evals",
                        lambda runs, *a, **k: seen.update(runs=runs, latent_off=k.get("latent_off")) or [])
    monkeypatch.setattr(mod, "run_missing_passk",
                        lambda runs, *a, **k: seen.update(runs=runs) or [])

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["eval_d2.py", *argv])
        with pytest.raises(SystemExit):
            mod.main()
        return [(r["model_size"], r["variant_id"]) for r in seen["runs"]], seen.get("latent_off")
    yield main
    sys.modules.pop("eval_d2", None)


def test_eval_d2_selects_the_3b_run_only(eval_d2):
    vid = "v3_learned_lambda-rp1-drboth-dsmath"
    runs, off = eval_d2("--baselines", "--sizes", "3b", "--exact", "--variants", vid)
    assert runs == [("3b", vid)] and not off
    runs, off = eval_d2("--baselines", "--sizes", "3b", "--exact", "--variants", vid, "--latent_off")
    assert runs == [("3b", vid)] and off
    runs, _ = eval_d2()                                        # default: new arms only, as before
    assert runs == [("1.5b", "v3_learned_lambda-rp1-drboth-dsmath-sl0.5")]
    runs, _ = eval_d2("--passk", "8", "--sizes", "1.5b", "--exact", "--variants", vid)
    assert runs == [("1.5b", vid)]


def write_eval(ckpt, vec, task="math", name="eval_results.json"):
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, name), "w") as f:
        json.dump({"metrics": {"total": len(vec)},
                   "results": [{"question": f"q{i}", "correct": bool(c)} for i, c in enumerate(vec)]}, f)


def test_analyse_scale_pairs_per_size_task_seed(tmp_path, monkeypatch):
    import analyse_scale as asc
    monkeypatch.setitem(asc.TEST_SIZE, "math", 10)
    e = tmp_path / "experiments"
    write_eval(str(e / "3b-v1_grpo_baseline-rp1-drboth-dsmath/checkpoint-937"), [1] * 5 + [0] * 5)
    write_eval(str(e / "3b-v3_learned_lambda-rp1-drboth-dsmath/checkpoint-937"), [1] * 8 + [0] * 2)
    write_eval(str(e / "3b-v3_learned_lambda-rp1-drboth-dsmath/checkpoint-937"), [1] * 6 + [0] * 4,
               name="eval_results_latent_off.json")
    write_eval(str(e / "3b-v1_grpo_baseline-rp1-drboth-dsmath/checkpoint-250"), [0] * 10)  # older
    write_eval(str(e / "1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath/checkpoint-937"), [1] * 10)
    write_eval(str(e / "new-3b-v3_learned_lambda-rp1-drboth-dsmath-lp0.2/checkpoint-937"), [1] * 10)
    rows, summary = asc.analyse(str(e))
    assert len(rows) == 1                                      # 1.5B seed 43 has no v1; new- ignored
    r = rows[0]
    assert (r["size"], r["task"], r["seed"]) == ("3b", "math", 42)
    assert (r["v1"], r["v3"], r["delta_pp"]) == (50.0, 80.0, 30.0)
    assert (r["v1_only"], r["v3_only"]) == (0, 3)
    assert r["off_cost_pp"] == pytest.approx(-20.0)
    assert summary == [{"size": "3b", "task": "math", "n_seeds": 1, "mean_delta_pp": 30.0,
                        "positive": 1, "mean_off_cost_pp": pytest.approx(-20.0), "n_off": 1}]
    write_eval(str(e / "3b-v3_learned_lambda-rp1-drboth-dsmath/checkpoint-937"), [1] * 10)
    with open(e / "3b-v3_learned_lambda-rp1-drboth-dsmath/checkpoint-937/eval_results.json") as f:
        data = json.load(f)
    data["results"][0]["question"] = "different"
    with open(e / "3b-v3_learned_lambda-rp1-drboth-dsmath/checkpoint-937/eval_results.json", "w") as f:
        json.dump(data, f)
    with pytest.raises(SystemExit):
        asc.analyse(str(e))
