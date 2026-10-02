"""
eval_passk.py - sampled evaluation: k completions per question, scored as avg@k,
pass@k and maj@k, plus the mean number of generated tokens.

Complements, never replaces, the greedy single-decode protocol of eval_gsm8k.py:
results go to eval_results_pass{k}.json beside eval_results.json.

Protocol
  * the checkpoint is loaded by eval_gsm8k.load_model_for_eval, i.e. with exactly the
    gate configuration the greedy evaluation uses;
  * each prompt is repeated k times in the batch and decoded with sampling
    (is_inference=False) at `temperature`. The driver passes the run's own TRAINING
    temperature: in this codebase one temperature variable drives both sampling and
    the Eq.3 latent, so that is the setting the latent channel was trained under.
    It is recorded in the metrics; runs trained at different temperatures are
    therefore sampled at different temperatures, which must be stated when they are
    compared;
  * torch is seeded with `sample_seed` before decoding so a re-run reproduces.

Cost: k times the greedy evaluation. batch_size counts QUESTIONS, so the model sees
batch_size * k sequences at once.
"""

import unsloth  # noqa: F401  (must be imported before transformers/trl pieces)

import json
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import GenerationConfig

from eval_gsm8k import get_task_spec, load_model_for_eval
from passk_utils import aggregate_samples, generated_length, summarise_questions
from utils import SYSTEM_PROMPT, extract_from_response


def evaluate_model_sampled(
    model_path: str,
    adapter_path: str,
    temperature: float,
    k: int = 8,
    batch_size: int = 8,
    num_samples: int = None,
    gate_mode: str = "standard",
    save_path: str = None,
    task: str = "gsm8k",
    sample_seed: int = 0,
):
    model, tokenizer = load_model_for_eval(model_path, adapter_path, gate_mode, task)
    dataset, q_key, a_key, max_new, answer_fn, gold_fn = get_task_spec(task)
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} questions; sampling k={k} at temperature {temperature}")

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    eos_id = tokenizer.eos_token_id
    eos_text = tokenizer.special_tokens_map['eos_token']

    per_question = []
    progress_bar = tqdm(total=total_samples, desc=f"pass@{k}", unit="q", dynamic_ncols=True)
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data[q_key])
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data[q_key]
        ]
        formatted_prompts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left",
            add_special_tokens=False,
        )
        # Row j*k + s is sample s of question j.
        prompt_ids = prompt_inputs["input_ids"].to(model.device).repeat_interleave(k, dim=0)
        prompt_mask = prompt_inputs["attention_mask"].to(model.device).repeat_interleave(k, dim=0)
        prompt_length = prompt_ids.size(1)

        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new,
            ),
            processing_class=tokenizer,
            is_inference=False,          # sample, do not take the argmax
        )

        for j in range(current_batch_size):
            true_answer = gold_fn(batch_data[a_key][j])
            answers, lengths = [], []
            for s in range(k):
                generated = outputs[j * k + s][prompt_length:]
                lengths.append(generated_length(generated.tolist(), eos_id))
                response = tokenizer.decode(generated).split(eos_text)[0]
                answers.append(answer_fn(extract_from_response(response)))
            scored = aggregate_samples(answers, true_answer)
            per_question.append({
                'question': batch_data[q_key][j],
                'true_answer': true_answer,
                'generated_answers': answers,
                'correct_flags': scored["correct_flags"],
                'avg_acc': scored["avg_acc"],
                'pass_at_k': scored["pass_at_k"],
                'maj_answer': scored["maj_answer"],
                'maj_at_k': scored["maj_at_k"],
                'mean_tokens': sum(lengths) / len(lengths),
            })

        progress_bar.update(current_batch_size)
        running = summarise_questions(per_question)
        progress_bar.set_postfix({'avg': f"{running['avg_at_k']*100:.2f}%",
                                  'pass': f"{running['pass_at_k']*100:.2f}%"})
    progress_bar.close()

    metrics = summarise_questions(per_question)
    metrics.update({
        'k': k,
        'temperature': temperature,
        'sample_seed': sample_seed,
        'model_path': adapter_path,
        'base_model': model_path,
        'gate_mode': gate_mode,
        'task': task,
        'timestamp': datetime.now().isoformat(),
    })
    if save_path is None:
        save_path = f"{adapter_path}/eval_results_pass{k}.json"
    with open(save_path, 'w') as f:
        json.dump({'metrics': metrics, 'results': per_question}, f, indent=2)
    print(f"\navg@{k}={metrics['avg_at_k']*100:.2f}  pass@{k}={metrics['pass_at_k']*100:.2f}  "
          f"maj@{k}={metrics['maj_at_k']*100:.2f}  tokens={metrics['mean_tokens']:.1f}  -> {save_path}")
    return metrics
