from __future__ import annotations

from typing import Any

from rollout.model_runner import Message


class TransformersGenerator:
    """Lazy optional adapter for local Hugging Face causal language models."""

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model evaluation requires torch and transformers. "
                "Install the packages in requirements-model.txt."
            ) from exc

        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self._torch = torch
        self._device = self._resolve_device(device)
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def __call__(self, messages: list[Message]) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        generation_args: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self._temperature > 0:
            generation_args["temperature"] = self._temperature

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_args)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
