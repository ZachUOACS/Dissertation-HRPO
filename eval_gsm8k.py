import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def load_model_for_eval(model_path: str, adapter_path: str, gate_mode: str = "standard",
                        task: str = "gsm8k", latent_temperature: float = None,
                        blend_gate: dict = None, weight_quant: str = None):
    """Load a base model (+ optional adapter) with the gate configuration it was
    trained with, ready for generation. Returns (model, tokenizer).

    This is the body evaluate_model always ran, moved out unchanged so that the
    sampled pass@k evaluation (eval_passk.py) loads checkpoints through the SAME
    code: a wrong gate mode makes an accuracy meaningless, so there must be one
    place that decides it. See evaluate_model for what each gate_mode means.

    latent_temperature: the Eq.3 mixture temperature the run was trained with
    (--latent_temperature); None (default) mixes over the sampling distribution,
    the original behaviour.
    blend_gate: the D3 settings the run was trained with, as a dict with keys
    blend_gate / blend_max / blend_d0 / blend_d1 (None = the variant's own blend).
    Ignored when gate_mode is "disabled" (latent-off), where nothing is blended.
    weight_quant: None (default) leaves the weights alone; otherwise a weight_quant.SCHEMES
    name: the LoRA adapters are merged and the block projections quantised in place, and the
    stats are left on model.weight_quant_stats."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        # MATH decodes 1024 new tokens on top of the prompt; 1024 total
        # would truncate. (eval_math.py uses 1536 for the same reason.)
        max_seq_length = 1536 if task in ("math", "mmlu") else 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # Same prompts as in training: Llama 3.2's template would otherwise stamp today's date into
    # every prompt (model_registry.pin_chat_template_date; no-op for Qwen).
    from model_registry import pin_chat_template_date
    pin_chat_template_date(tokenizer)

    u_init = None
    if gate_mode == "constant":
        # u must exist on thinking_residual_Lambda BEFORE the adapter is loaded,
        # exactly as in training (register_constant_latent_vector precedes
        # get_peft_model). Register it afterwards and the checkpoint's u has no
        # parameter to land in: it is dropped, u stays at its init, and the run
        # is evaluated with a vector it never trained. Keep the init value so
        # the load can be verified below.
        from variant_common import register_constant_latent_vector, get_inner_model
        register_constant_latent_vector(model)
        u_init = get_inner_model(model).thinking_residual_Lambda.u.detach().clone()

    if adapter_path is not None:
        model.load_adapter(adapter_path)
    if gate_mode == "disabled":
        model.answer_start = ""
    elif gate_mode == "unit":
        from variant_common import apply_unit_gate_patch
        apply_unit_gate_patch(model)
    elif gate_mode == "constant":
        from variant_common import apply_constant_latent_patch, get_inner_model
        apply_constant_latent_patch(model)
        if adapter_path is not None:
            # A silently-dropped u is the failure this whole branch exists to
            # prevent, and it is invisible in the accuracy number — so make it
            # loud. Training moves u (it sits in Lambda's 1e-3 group), so a
            # bit-identical u means the checkpoint's value never arrived.
            u_now = get_inner_model(model).thinking_residual_Lambda.u.detach()
            if torch.equal(u_now, u_init.to(u_now.device)):
                raise RuntimeError(
                    f"gate_mode='constant': u is bit-identical to its init after "
                    f"loading {adapter_path} — the checkpoint's learned u was not "
                    f"applied, so this evaluation would use the wrong latent "
                    f"vector. Check that the adapter carries a "
                    f"'thinking_residual_Lambda...u' key.")
    elif gate_mode != "standard":
        raise ValueError(f"Unknown gate_mode: {gate_mode}")
    if latent_temperature is not None:
        from variant_common import set_latent_temperature
        set_latent_temperature(model, latent_temperature)
    if blend_gate and gate_mode != "disabled":
        # after the gate-mode patch above, which also overrides thinking_residual
        from variant_common import set_blend_gate
        set_blend_gate(model, **blend_gate)
    quant_stats = None
    if weight_quant is not None:
        # before for_inference, so nothing it prepares can hold the unquantised weights
        from weight_quant import apply_weight_quant
        quant_stats = apply_weight_quant(model, weight_quant)
        print(f"[weight_quant] {quant_stats}")
    model = FastLanguageModel.for_inference(model)
    if quant_stats is not None:
        model.weight_quant_stats = quant_stats
    return model, tokenizer


def get_task_spec(task: str):
    """(dataset, q_key, a_key, max_new, answer_fn, gold_fn) for a task.

    GSM8K and MATH share the system prompt, the "####" answer marker and
    extract_from_response; they differ only in the split, the field holding the
    gold answer, how that field becomes a comparable answer, and the decode
    budget (MATH solutions are longer). Moved out of evaluate_model unchanged so
    eval_passk.py scores with the same comparators."""
    if task == "gsm8k":
        dataset = load_dataset('openai/gsm8k', 'main')['test']
        q_key, a_key, max_new = 'question', 'answer', 512
        answer_fn = process_gsm8k_answer
        gold_fn = lambda gold: process_gsm8k_answer(extract_hash_answer(gold))
    elif task == "math":
        # MATH-500 (or a local MATH test split): 'problem' + boxed 'solution'.
        dataset = load_math500()
        q_key, a_key, max_new = 'problem', 'solution', 1024
        answer_fn = process_math_answer
        gold_fn = lambda gold: process_math_answer(extract_boxed_answer(gold) or "")
    elif task == "mmlu":
        # Cross-model study: MMLU-STEM test + ARC-C test as one set (mmlu_data.load_test_set), the
        # prompt text built exactly as in training, the gold a letter, each answer's source kept.
        from mmlu_data import load_test_set
        dataset = load_test_set()
        q_key, a_key, max_new = 'prompt_text', 'answer_letter', 512
        answer_fn = process_mmlu_answer
        gold_fn = lambda gold: gold
    else:
        raise ValueError(f"unknown task {task!r}; expected 'gsm8k', 'math' or 'mmlu'")
    return dataset, q_key, a_key, max_new, answer_fn, gold_fn


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    gate_mode: str = "standard",
    save_path: str = None,
    task: str = "gsm8k",
    latent_temperature: float = None,
    blend_gate: dict = None,
    weight_quant: str = None,
):
    """gate_mode selects the gate configuration used at inference and must
    mirror how the checkpoint was trained (see changes.md mapping #7):
      - "standard": stock HRPO gate (v4/v5/v6 and the legacy checkpoints);
      - "unit":     r_t = i_t = 1 via the unit-gate patch (v2/v3);
      - "constant": unit gates AND the Eq.3 projection replaced by the learned
                    vector u (v7 / tier-8 arm C1). Without this, a C1
                    checkpoint would be evaluated with the projection it was
                    never trained with and its accuracy would be meaningless;
      - "disabled": no hybrid blending at all (v1 and base zero-shot —
                    without an HRPO-trained adapter the gate weights are
                    random and must not run).
    adapter_path=None evaluates the base model zero-shot (save_path required
    if save_results). Defaults reproduce the original behaviour exactly.
    weight_quant (see load_model_for_eval) also records each answer's generated token
    count ('num_tokens', up to the first end-of-sequence token) and their mean.
    """
    model, tokenizer = load_model_for_eval(model_path, adapter_path, gate_mode, task,
                                           latent_temperature=latent_temperature,
                                           blend_gate=blend_gate, weight_quant=weight_quant)
    dataset, q_key, a_key, max_new, answer_fn, gold_fn = get_task_spec(task)
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data[q_key])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data[q_key]
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=max_new,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = answer_fn(extracted)
            true_answer = gold_fn(batch_data[a_key][j])
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                'question': batch_data[q_key][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            if 'source' in batch_data:          # mmlu: which test the question is from
                result['source'] = batch_data['source'][j]
            if weight_quant is not None:
                generated = output[prompt_length:]
                ends = (generated == tokenizer.eos_token_id).nonzero()
                result['num_tokens'] = int(ends[0, 0]) if len(ends) else int(generated.numel())
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': adapter_path,
        'base_model': model_path,
        'gate_mode': gate_mode,
        'task': task,
        'timestamp': datetime.now().isoformat()
    }
    if latent_temperature is not None:
        # only recorded when set, so every default results file keeps its old keys
        metrics['latent_temperature'] = latent_temperature
    if blend_gate and gate_mode != "disabled":
        metrics['blend_gate'] = blend_gate
    if weight_quant is not None:
        metrics['weight_quant'] = model.weight_quant_stats
        metrics['mean_tokens'] = (sum(r['num_tokens'] for r in results) / len(results)
                                  if results else None)

    if save_results:
        if save_path is None:
            save_path = adapter_path + "/eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        if model.split('/')[-1] in checkpoint_path:
            base_model = model
    temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
    print(checkpoint_path, base_model, temperature)

    if 'eval_results.json' not in os.listdir(checkpoint_path):
        print(f"Starting GSM8k evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
        )