from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout.model_runner import SYSTEM_PROMPT


IGNORE_INDEX = -100


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in resolve_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def assistant_action_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in row["messages"]:
            if message["role"] == "assistant":
                examples.append(
                    {
                        "id": row["id"],
                        "prompt_messages": list(conversation),
                        "target": message["content"],
                    }
                )
            conversation.append(message)
    return examples


class NextActionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
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
            keep = self.max_seq_len
            input_ids = input_ids[-keep:]
            labels = labels[-keep:]
            if all(label == IGNORE_INDEX for label in labels):
                labels[-len(target_ids) :] = target_ids[-keep:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        input_ids = []
        labels = []
        attention_mask = []
        for item in features:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                torch.nn.functional.pad(
                    torch.ones_like(item["input_ids"], dtype=torch.long),
                    (0, pad_len),
                    value=0,
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


def build_model(model_path: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> Any:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.train_data, limit=args.limit_rows)
    examples = assistant_action_examples(rows)
    dataset = NextActionDataset(examples, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer.pad_token_id),
    )
    model = build_model(args.model, args.lora_r, args.lora_alpha, args.lora_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    logs: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    max_optimizer_steps = args.max_steps
    total_batches = len(loader) * args.epochs
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            if torch.cuda.is_available():
                batch = {key: value.cuda() for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                avg_loss = running_loss / args.gradient_accumulation_steps
                running_loss = 0.0
                if optimizer_step % args.log_every == 0 or optimizer_step == 1:
                    log = {
                        "epoch": epoch + 1,
                        "optimizer_step": optimizer_step,
                        "global_step": global_step,
                        "loss": avg_loss,
                    }
                    logs.append(log)
                    print(json.dumps(log, ensure_ascii=False))
                if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
                    break
        if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
            break

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model": args.model,
        "train_data": args.train_data,
        "rows": len(rows),
        "next_action_examples": len(examples),
        "epochs": args.epochs,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer_steps": optimizer_step,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "logs": logs,
        "total_batches_available": total_batches,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for WebNav next-action tool calls.")
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--train-data", default="training/sft_train.jsonl")
    parser.add_argument("--output-dir", default="outputs/checkpoints/qwen2_5_0_5b_lora_sft")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional optimizer step cap for smoke runs.")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metadata = train(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
