"""Cross-model study plumbing (../CROSS_MODEL_PLAN.md): model_registry.py, the size keys in every
directory regex, smoke-run isolation (--max_steps), submit_d2.sh family, run_ablations.sh and the
smoke job.

What must hold:
  * every size key is a single directory token and SIZE_ALT matches exactly the registered keys;
  * a Llama baseline is a plain baseline directory whose id is what the job scores, and a
    treatment is a new-* directory with the right axis token, compared against the same-size v3;
  * every directory regex (training, eval_d2, analyse_d2, analyse_quant, analyse_scale) accepts the
    new keys, and none of them ever matches a smoke-* directory;
  * the family tier submits one job per (model, task, arm, seed), with AUTO_EVAL_BASELINE set for
    baselines only, the right wall times, and arguments identical to the Qwen baselines';
  * the smoke job and smoke_family.py describe the same arms.

    python -m pytest tests/test_family.py -q
"""
import importlib
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

import model_registry as mr  # noqa: E402

NEW_KEYS = ("llama1b", "llama3b")
BASH = shutil.which("bash")


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- registry and names

def test_registry_keys():
    assert {"0.5b", "1.5b", "3b"} <= set(mr.MODEL_NAMES)
    assert mr.MODEL_NAMES["llama1b"] == "unsloth/Llama-3.2-1B-Instruct"
    assert mr.MODEL_NAMES["llama3b"] == "unsloth/Llama-3.2-3B-Instruct"
    assert "gemma2b" not in mr.MODEL_NAMES, "Gemma needs the rollout port first"
    for k in mr.MODEL_NAMES:
        assert "-" not in k and re.fullmatch(mr.SIZE_ALT, k)
        assert k in mr.FAMILY
    for bad in ("llama", "1b", "b", "llama1b-x", "3"):
        assert not re.fullmatch(mr.SIZE_ALT, bad)


def test_variant_common_reexports_registry():
    import variant_common as vc
    assert vc.MODEL_NAMES is mr.MODEL_NAMES


@pytest.mark.parametrize("size", NEW_KEYS)
@pytest.mark.parametrize("task", ["gsm8k", "math"])
@pytest.mark.parametrize("seed", ["42", "43"])
def test_llama_directories(size, task, seed):
    import variant_common as vc
    base = ["--repetition_penalty", "1.0", "--dr_grpo", "both", "--seed", seed]
    if task == "math":
        base += ["--dataset", "math", "--max_completion_length", "1024"]
    for variant in ("v1_grpo_baseline", "v3_learned_lambda", "v4_full_hrpo_gate"):
        name = os.path.basename(vc.experiment_dir(size, variant, vc.build_arg_parser(variant).parse_args(base)))
        assert name.startswith(f"{size}-{variant}") and not name.startswith("new-")
        assert vc.parse_experiment_name(name)[0] == size
    for extra, token in ((["--stochastic_latent", "0.5"], "-sl0.5"), (["--latent_temperature", "1.0"], "-lt1")):
        args = vc.build_arg_parser("v3_learned_lambda").parse_args(base + extra)
        name = os.path.basename(vc.experiment_dir(size, "v3_learned_lambda", args))
        assert name.startswith(f"new-{size}-v3_learned_lambda") and name.endswith(token)


@pytest.fixture
def eval_d2(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
    stub = types.ModuleType("eval_all")
    stub.TASK_TEST_SIZE = {"gsm8k": 1319, "math": 500}
    for n in ("discover_experiments", "final_checkpoint", "get_run_temperature", "run_missing_evals"):
        setattr(stub, n, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "eval_all", stub)
    sys.modules.pop("eval_d2", None)
    return importlib.import_module("eval_d2")


def test_every_directory_regex_takes_new_keys_and_skips_smoke(eval_d2):
    import variant_common as vc
    import analyse_d2
    import analyse_quant
    import analyse_scale
    base = ["--repetition_penalty", "1.0", "--dr_grpo", "both", "--seed", "43", "--dataset", "math"]
    for size in NEW_KEYS:
        baseline = f"{size}-v3_learned_lambda-seed43-rp1-drboth-dsmath"
        treat = f"new-{size}-v3_learned_lambda-seed43-rp1-drboth-dsmath-sl0.5"
        assert vc.EXPERIMENT_NAME_RE.match(baseline)
        assert analyse_d2.BASE_RE.match(baseline).group("size") == size
        assert analyse_d2.split_name(treat)[0] == size
        assert analyse_quant.DIR_RE.match(baseline) and analyse_quant.DIR_RE.match(treat)
        assert analyse_scale.DIR_RE.match(baseline).group("size") == size
        assert eval_d2.NEW_RE.match(treat).group("size") == size
        # Smoke runs: a real-looking name with the smoke prefix, from the real naming code.
        for extra in ([], ["--stochastic_latent", "0.5"]):
            args = vc.build_arg_parser("v3_learned_lambda").parse_args(base + extra + ["--max_steps", "4"])
            smoke = os.path.basename(vc.experiment_dir(size, "v3_learned_lambda", args))
            assert smoke.startswith("smoke-") and smoke.endswith("-steps4")
            for rx in (vc.EXPERIMENT_NAME_RE, analyse_d2.BASE_RE, analyse_quant.DIR_RE,
                       analyse_scale.DIR_RE, eval_d2.NEW_RE):
                assert not rx.match(smoke), (rx.pattern, smoke)
            assert analyse_d2.split_name(smoke) is None


def test_qwen_names_unchanged():
    import variant_common as vc
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(
        ["--repetition_penalty", "1.0", "--dr_grpo", "both", "--seed", "43", "--dataset", "math"])
    assert os.path.basename(vc.experiment_dir("1.5b", "v3_learned_lambda", args)) == \
        "1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath"
    assert args.max_steps == -1


def test_analyse_scale_orders_sizes_by_registry():
    import analyse_scale
    assert analyse_scale.SIZE_ORDER("1.5b") < analyse_scale.SIZE_ORDER("3b") < analyse_scale.SIZE_ORDER("llama1b")


def test_max_steps_reaches_the_trainer():
    src = read("variant_common.py")
    body = src[src.index("def train_with_resume"):src.index("trainer.train(resume_from_checkpoint=resume)")]
    assert "trainer.args.max_steps = args.max_steps" in body
    assert "trainer.args.save_steps = args.max_steps" in body


# --------------------------------------------------------------------------- shell scripts

def bash_fn(src, fn, *argv):
    if BASH is None:
        pytest.skip("bash not available")
    body = re.search(rf"^{fn}\(\) \{{.*?^\}}", src, re.M | re.S).group(0)
    return subprocess.run([BASH, "-c", body + f'\n{fn} "$@"', "x", *argv],
                          capture_output=True, text=True)


@pytest.mark.parametrize("size", NEW_KEYS)
def test_run_ablations_accepts_new_keys(size):
    out = bash_fn(read("run_ablations.sh"), "variants_for_size", size)
    assert out.returncode == 0 and out.stdout.split() == ["v1", "v3", "v4"]


def test_baseline_id_v4():
    out = bash_fn(read("cluster/submit_d2.sh"), "baseline_id", "4", "43", "math")
    assert out.stdout.strip() == "v4_full_hrpo_gate-seed43-rp1-drboth-dsmath"


def dry_run(tmp_path, **env):
    """Run the family tier in submit_d2.sh's dry-run mode: with SUBMIT_DRY_RUN set, submit() appends
    each job to that file and never calls sbatch.

    Never test with a fake sbatch on the PATH. On 23 October this test's fake lived in /tmp, which the
    cluster mounts noexec; bash's PATH search skipped it and the real sbatch submitted 40 jobs. As a
    second guard, sbatch is also exported as a bash function that refuses (functions take precedence
    over the PATH), so even a broken dry-run branch cannot reach the real one."""
    if BASH is None:
        pytest.skip("bash not available")
    log = tmp_path / "jobs.txt"
    script = os.path.join(ROOT, "cluster", "submit_d2.sh")
    guard = 'sbatch() { echo "REAL SBATCH BLOCKED BY TEST" >&2; return 99; }; export -f sbatch; '
    cmd = guard + f'bash "$(cygpath -u "{script}" 2>/dev/null || echo "{script}")" family'
    out = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, SUBMIT_DRY_RUN=log.as_posix(), **env))
    assert "REAL SBATCH BLOCKED" not in out.stderr, out.stderr
    assert not re.search(r"-> job \d", out.stdout), out.stdout
    jobs = []
    if log.exists():
        for line in log.read_text().splitlines():
            argv, auto, extra = line.split("\t")
            jobs.append({"argv": argv, "auto": auto, "extra": extra})
    assert out.stdout.count("-> job DRYRUN") == len(jobs), out.stdout
    return out, jobs


def job_field(argv, flag):
    m = re.search(rf"{flag}=(\S+)", argv)
    return m.group(1) if m else None


def test_family_tier_dry_run(tmp_path):
    import variant_common as vc
    out, jobs = dry_run(tmp_path, FAM_MODELS="llama1b llama3b", FAM_TASKS="gsm8k math",
                        FAM_ARMS="v1 v3 v4 c1 lt", FAM_SEEDS="42 43")
    assert out.returncode == 0, out.stderr
    assert len(jobs) == 2 * 2 * 5 * 2
    variants = {"1": "v1_grpo_baseline", "3": "v3_learned_lambda", "4": "v4_full_hrpo_gate"}
    for j in jobs:
        name = job_field(j["argv"], "--job-name")
        size = re.search(r"MODEL_SIZE=([a-z0-9.]+)", j["argv"]).group(1)
        vnum = re.search(r"VARIANT_NUM=(\d)", j["argv"]).group(1)
        assert name.startswith(f"hrpo-fam-{size}-") and size in NEW_KEYS
        assert job_field(j["argv"], "--mem") == "32G"
        task = "math" if "--dataset math" in j["extra"] else "gsm8k"
        want = {("llama3b", "math"): "96:00:00", ("llama1b", "math"): "60:00:00",
                ("llama3b", "gsm8k"): "48:00:00", ("llama1b", "gsm8k"): "24:00:00"}[(size, task)]
        assert job_field(j["argv"], "--time") == want
        args = vc.build_arg_parser(variants[vnum]).parse_args(j["extra"].split())
        dirname = os.path.basename(vc.experiment_dir(size, variants[vnum], args))
        if "-sl0.5-" in name or "-lt1.0-" in name:
            assert j["auto"] == "" and dirname.startswith(f"new-{size}-v3_learned_lambda")
        else:
            assert j["auto"] and dirname == f"{size}-{j['auto']}"


def test_family_tier_matches_qwen_baseline_arguments():
    src = read("cluster/submit_d2.sh")
    fam = src[src.index("    family)"):src.index("    *)\n", src.index("    family)"))]
    s3b = src[src.index("    scale3b)"):src.index("    family)")]
    grab = lambda t, n: " ".join(re.findall(rf'^\s*{n}="([^"]*)"', t, re.M))  # noqa: E731
    for n in ("GSM8K_ARGS", "MATH_ARGS"):
        assert grab(fam, n) == grab(s3b, n)


def test_family_tier_rejects_unknown_model(tmp_path):
    out, jobs = dry_run(tmp_path, FAM_MODELS="gemma2b", FAM_ARMS="v1", FAM_SEEDS="42")
    assert out.returncode != 0 and not jobs


def test_smoke_job_and_checker_agree():
    slurm = read("cluster/run_smoke_family.slurm")
    rows = re.findall(r'^\s*"([a-z0-9]+)\|([a-z_]+\.py)\|([a-z0-9]+)\|([^"]*)"', slurm, re.M)
    src = read("smoke_family.py")
    block = src[src.index("ARMS = {"):src.index("}\n", src.index("ARMS = {"))]
    checker = {m[0]: (m[1], m[2], m[3].replace('"', "").replace(",", "").split())
               for m in re.findall(r'"([a-z0-9]+)": \("([a-z0-9_]+)", "([a-z0-9]+)", \[([^\]]*)\]', block)}
    scripts = {"v1_grpo_baseline": "hrpo_grpo_baseline.py", "v3_learned_lambda": "hrpo_learned_lambda.py",
               "v4_full_hrpo_gate": "hrpo_full_gate.py"}
    assert {r[0] for r in rows} == set(checker)
    for arm, script, task, extra in rows:
        variant, ctask, cextra = checker[arm]
        assert scripts[variant] == script and ctask == task and cextra == extra.split()


def test_run_variant_scores_family_treatments():
    src = read("cluster/run_variant.slurm")
    case = src[src.index('case " ${EXTRA_ARGS:-} " in'):src.index("esac")]
    assert '" --stochastic_latent "' in case and '" --latent_temperature "' in case


def test_eval_tokens_uses_each_runs_tokeniser():
    import eval_tokens as et
    assert et.tokenizer_name("experiments/1.5b-v1_grpo_baseline-rp1-drboth") == et.TOKENIZER
    assert et.tokenizer_name("experiments/3b-v3_learned_lambda-rp1-drboth-dsmath") == et.TOKENIZER
    assert et.tokenizer_name("experiments/llama1b-v1_grpo_baseline-rp1-drboth") == mr.MODEL_NAMES["llama1b"]
    assert et.tokenizer_name("experiments/new-llama3b-v3_learned_lambda-rp1-drboth-sl0.5") == \
        mr.MODEL_NAMES["llama3b"]


def test_chat_template_date_is_pinned():
    jinja2 = pytest.importorskip("jinja2")
    llama_like = ('{%- if not date_string is defined %}{%- if strftime_now is defined %}'
                  '{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}'
                  '{%- set date_string = "26 Jul 2024" %}{%- endif %}{%- endif %}'
                  'Today Date: {{ date_string }}')
    tok = types.SimpleNamespace(chat_template=llama_like)
    assert mr.pin_chat_template_date(tok) and not mr.pin_chat_template_date(tok)   # idempotent
    env = jinja2.Environment()
    env.globals["strftime_now"] = lambda fmt: "01 Jan 2030"
    assert env.from_string(llama_like).render() == "Today Date: 01 Jan 2030"
    assert env.from_string(tok.chat_template).render().strip() == f"Today Date: {mr.PINNED_DATE}"
    qwen_like = types.SimpleNamespace(chat_template="{{ messages }}<|im_start|>")
    assert not mr.pin_chat_template_date(qwen_like) and qwen_like.chat_template == "{{ messages }}<|im_start|>"
    assert "pin_chat_template_date(tokenizer)" in read("eval_gsm8k.py")
    assert "pin_chat_template_date(getattr(trainer, \"processing_class\", None))" in read("variant_common.py")



def test_submit_has_a_dry_run_that_never_calls_sbatch():
    src = read("cluster/submit_d2.sh")
    body = src[src.index("submit() {"):src.index("\n}\n", src.index("submit() {"))]
    dry = body[body.index('if [[ -n "${SUBMIT_DRY_RUN:-}" ]]'):body.index("    else\n")]
    code = [ln for ln in dry.splitlines() if not ln.strip().startswith("#")]
    assert code and not any("sbatch" in ln for ln in code)
    # The smoke job runs unattended on a compute node: no tests there.
    assert "pytest" not in read("cluster/run_smoke_family.slurm")


# --------------------------------------------------------------------------- third task: mmlu

def test_registry_tasks_and_smollm2():
    assert mr.MODEL_NAMES["smollm2"] == "HuggingFaceTB/SmolLM2-1.7B-Instruct" and mr.FAMILY["smollm2"] == "smollm2"
    assert mr.TASK_TEST_SIZE == {"gsm8k": 1319, "math": 500, "mmlu": 4325}
    assert [mr.task_from_name(n) for n in ("x-rp1-drboth", "x-rp1-drboth-dsmath", "x-rp1-drboth-dsmmlu-sl0.5")] \
        == ["gsm8k", "math", "mmlu"]
    assert [mr.baseline_tag(t) for t in ("gsm8k", "math", "mmlu")] == ["drboth", "drboth-dsmath", "drboth-dsmmlu"]


def test_mmlu_rows_and_prompt_match_training():
    import mmlu_data as md
    import utils
    aux = md.from_mmlu_aux({"train": {"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": 2}})
    assert aux == {"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": 2, "source": "mmlu_aux"}
    arc = md.from_arc({"question": "Q2", "choices": {"text": ["x", "y", "z"], "label": ["1", "2", "3"]},
                       "answerKey": "3"})
    assert arc["answer"] == 2 and arc["choices"] == ["x", "y", "z"]
    arc_letters = md.from_arc({"question": "Q3", "choices": {"text": ["p", "q"], "label": ["A", "B"]},
                               "answerKey": "B"})
    assert arc_letters["answer"] == 1
    batch = utils.process_mmlu({"question": ["Q1"], "choices": [["a", "b", "c", "d"]], "answer": [2]})
    assert batch["prompt"][0][1]["content"] == md.format_prompt("Q1", ["a", "b", "c", "d"])
    assert batch["answer"] == ["C"]
    assert utils.process_mmlu_answer("so the answer is (C).") == "C"


def test_mmlu_test_set_order_and_columns(monkeypatch):
    datasets = pytest.importorskip("datasets")
    import mmlu_data as md
    fake = {
        md.MMLU_STEM_REPO: {"test": [{"question": "s1", "choices": ["a", "b"], "subject": "x", "answer": 1}]},
        md.ARC_REPO: {"test": [{"question": "c1", "choices": {"text": ["u", "v", "w"], "label": ["A", "B", "C"]},
                                "answerKey": "A"}]},
    }
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, cfg=None: fake[repo])
    ds = md.load_test_set()
    assert ds["source"] == ["mmlu_stem", "arc_c"] and ds["answer_letter"] == ["B", "A"]
    assert ds["prompt_text"][0] == md.format_prompt("s1", ["a", "b"])


def test_mmlu_training_dataset_and_reward(monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    import mmlu_data as md
    import variant_common as vc
    root = tmp_path / "mmlu_subset"
    datasets.Dataset.from_list([md.from_mmlu_aux({"train": {"question": "Q", "choices": ["a", "b"], "answer": 1}})]
                               ).save_to_disk(str(root))
    monkeypatch.setattr(md, "TRAIN_ROOT", str(root))
    ds = vc.build_train_dataset("mmlu")
    assert ds[0]["answer"] == "B" and ds[0]["prompt"][1]["content"] == md.format_prompt("Q", ["a", "b"])
    import utils
    assert vc.reward_answer_fn("mmlu") is utils.process_mmlu_answer
    monkeypatch.setattr(md, "TRAIN_ROOT", str(tmp_path / "missing"))
    with pytest.raises(SystemExit):
        vc.build_train_dataset("mmlu")


def test_mmlu_directories_and_analysis(eval_d2):
    import variant_common as vc
    import analyse_d2
    import analyse_scale
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(
        ["--dataset", "mmlu", "--max_prompt_length", "1024", "--repetition_penalty", "1.0", "--dr_grpo", "both",
         "--seed", "43"])
    name = os.path.basename(vc.experiment_dir("llama1b", "v3_learned_lambda", args))
    assert name == "llama1b-v3_learned_lambda-seed43-rp1-drboth-dsmmlu"
    m = analyse_scale.DIR_RE.match(name)
    assert m and mr.task_from_name(m.group("ds")) == "mmlu"
    assert "drboth-dsmmlu" in analyse_d2.BASELINE_TAGS
    assert analyse_d2.TASK_FULL["mmlu"] == 4325
    src = read("eval_d2.py")
    assert 'tuple("-rp1-" + baseline_tag(t) for t in TASK_SUFFIX)' in src


def test_family_tier_mmlu(tmp_path):
    have_data = os.path.isdir(os.path.join(ROOT, "data", "mmlu_arc_train_7473"))
    out, jobs = dry_run(tmp_path, FAM_MODELS="llama1b 1.5b", FAM_TASKS="mmlu", FAM_ARMS="v1 v3", FAM_SEEDS="42 43")
    if not have_data:
        assert out.returncode != 0 and not jobs and "prepare_mmlu_arc.py" in out.stderr
        return
    assert out.returncode == 0 and len(jobs) == 8
    for j in jobs:
        assert "--dataset mmlu" in j["extra"] and "--max_prompt_length 1024" in j["extra"]
        assert j["auto"].endswith("-rp1-drboth-dsmmlu")
