from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import load_jsonl
from training.sft_train import DataCollator, IGNORE_INDEX, resolve_path


def assistant_step_examples(records: list[dict[str, Any]], min_abs_advantage: float = 1e-8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        advantage = float(record["advantage"])
        if abs(advantage) < min_abs_advantage:
            continue
        messages = record["trajectory"]["messages"]
        conversation: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "assistant":
                examples.append(
                    {
                        "group_id": record["group_id"],
                        "task_id": record["task_id"],
                        "sample_index": record["sample_index"],
                        "advantage": advantage,
                        "prompt_messages": list(conversation),
                        "target": message["content"],
                        "reward": record["reward"]["total_reward"],
                    }
                )
            conversation.append(message)
    return examples


class AdvantageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt = self.tokenizer.apply_chat_template(
            example["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        target = example["target"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]
            labels = labels[-self.max_seq_len :]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "advantage": torch.tensor(float(example["advantage"]), dtype=torch.float32),
        }


@dataclass
class AdvantageCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        base = DataCollator(self.pad_token_id)(
            [{"input_ids": item["input_ids"], "labels": item["labels"]} for item in features]
        )
        base["advantage"] = torch.stack([item["advantage"] for item in features])
        return base


def token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(IGNORE_INDEX)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return selected_log_probs * mask, mask


def sequence_policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    clip_advantage: float,
) -> torch.Tensor:
    selected_log_probs, mask = token_log_probs(logits, labels)
    token_loss = -selected_log_probs
    token_counts = mask.sum(dim=1).clamp_min(1)
    sequence_ce = token_loss.sum(dim=1) / token_counts
    clipped_advantages = advantages.clamp(min=-clip_advantage, max=clip_advantage)
    return (sequence_ce * clipped_advantages).mean()


def sequence_kl_penalty(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_log_probs, mask = token_log_probs(policy_logits, labels)
    with torch.no_grad():
        reference_log_probs, _ = token_log_probs(reference_logits, labels)
    token_counts = mask.sum(dim=1).clamp_min(1)
    # Compute the cancellation-sensitive k3 estimator in fp32 even when the
    # models run in fp16. Tiny negative values otherwise appear from rounding.
    log_ratio = (reference_log_probs - policy_log_probs).float()
    token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    token_kl = token_kl.clamp_min(0.0)
    sequence_kl = (token_kl * mask).sum(dim=1) / token_counts
    sequence_policy_log_prob = policy_log_probs.sum(dim=1) / token_counts
    sequence_reference_log_prob = reference_log_probs.sum(dim=1) / token_counts
    return sequence_kl.mean(), sequence_policy_log_prob.mean(), sequence_reference_log_prob.mean()


def build_policy_model(model_path: str, adapter_path: str) -> Any:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=False)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    return model


def build_reference_model(model_path: str, adapter_path: str) -> Any:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=False)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    records = load_jsonl(args.rollouts)
    examples = assistant_step_examples(records)
    if args.limit_examples is not None:
        examples = examples[: args.limit_examples]
    if not examples:
        raise ValueError("No non-zero advantage examples found in rollout file.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = AdvantageDataset(examples, tokenizer, args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AdvantageCollator(tokenizer.pad_token_id),
    )
    model = build_policy_model(args.model, args.adapter)
    reference_model = None
    if args.kl_beta > 0:
        reference_model = build_reference_model(args.model, args.reference_adapter or args.adapter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs: list[dict[str, Any]] = []
    optimizer_step = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in loader:
            if torch.cuda.is_available():
                batch = {key: value.cuda() for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=None,
            )
            policy_loss = sequence_policy_loss(
                outputs.logits,
                batch["labels"],
                batch["advantage"],
                clip_advantage=args.clip_advantage,
            )
            kl_loss = torch.zeros((), device=outputs.logits.device)
            mean_policy_log_prob = torch.zeros((), device=outputs.logits.device)
            mean_reference_log_prob = torch.zeros((), device=outputs.logits.device)
            if reference_model is not None:
                with torch.no_grad():
                    reference_outputs = reference_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=None,
                    )
                kl_loss, mean_policy_log_prob, mean_reference_log_prob = sequence_kl_penalty(
                    outputs.logits,
                    reference_outputs.logits,
                    batch["labels"],
                )
            loss = policy_loss + args.kl_beta * kl_loss
            (loss / args.gradient_accumulation_steps).backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % args.log_every == 0 or optimizer_step == 1:
                    log = {
                        "epoch": epoch + 1,
                        "optimizer_step": optimizer_step,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "kl_loss": float(kl_loss.detach().cpu()),
                        "kl_beta": args.kl_beta,
                        "mean_advantage": float(batch["advantage"].detach().float().mean().cpu()),
                        "mean_policy_log_prob": float(mean_policy_log_prob.detach().cpu()),
                        "mean_reference_log_prob": float(mean_reference_log_prob.detach().cpu()),
                    }
                    logs.append(log)
                    print(json.dumps(log, ensure_ascii=False))
                if args.max_steps is not None and optimizer_step >= args.max_steps:
                    break
        if args.max_steps is not None and optimizer_step >= args.max_steps:
            break

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "reference_adapter": args.reference_adapter or args.adapter,
        "rollouts": args.rollouts,
        "output_dir": args.output_dir,
        "records": len(records),
        "nonzero_advantage_examples": len(examples),
        "epochs": args.epochs,
        "optimizer_steps": optimizer_step,
        "global_steps": global_step,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "clip_advantage": args.clip_advantage,
        "kl_beta": args.kl_beta,
        "kl_estimator": "k3 on sampled target tokens: exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1",
        "logs": logs,
    }
    (output_dir / "grpo_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO-style LoRA update from grouped rollout records.")
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200")
    parser.add_argument("--reference-adapter", default=None, help="Frozen reference adapter. Defaults to --adapter.")
    parser.add_argument("--rollouts", default="outputs/rollouts/grpo_sft_step200_group4_task4.jsonl")
    parser.add_argument("--output-dir", default="outputs/checkpoints/qwen2_5_0_5b_lora_grpo_proto")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--clip-advantage", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metadata = train(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
