"""Provider configuration, multimodal request construction, and JSON parsing."""

from __future__ import annotations

import base64
import json
import mimetypes
import math
import os
import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """Analyse the document image using only visible evidence.
Return one valid JSON object with no surrounding prose. Do not guess unreadable values.
Use null for missing scalar fields. Return this shape:
{
  "document_type": "string or null",
  "fields": {"requested_field": "value or null"},
  "answer": "string",
  "evidence": [{"field": "field name", "text": "visible snippet"}],
  "uncertainties": ["description"],
  "confidence": 0.0
}
"""


class OutputParseError(ValueError):
    pass


def parse_with_raw(raw_text, expected_fields):
    try:
        return parse_model_output(raw_text, expected_fields)
    except OutputParseError as exc:
        exc.raw_output = raw_text
        raise


def parse_model_output(raw_text: str, expected_fields: list[str] | None = None) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise OutputParseError("The model returned an empty response.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    text = fenced.group(1) if fenced else raw_text.strip()
    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise OutputParseError("The response did not contain a valid JSON object.")

    if "fields" in parsed and not isinstance(parsed["fields"], dict):
        raise OutputParseError("The fields property must be an object.")
    fields = parsed.get("fields", {})
    # Some vision-language models follow the requested field schema directly
    # and return the extracted fields at the top level. Accept that useful
    # output instead of incorrectly scoring every requested field as missing.
    if "fields" not in parsed and expected_fields and any(k in parsed for k in expected_fields):
        fields = {name: parsed.get(name) for name in expected_fields}
    elif "fields" not in parsed:
        raise OutputParseError("No requested field object was found.")
    if expected_fields:
        fields = {name: fields.get(name) for name in expected_fields}
    try:
        confidence = float(parsed.get("confidence"))
        confidence = max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "document_type": parsed.get("document_type"),
        "fields": fields,
        "answer": str(parsed.get("answer") or ""),
        "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
        "uncertainties": parsed.get("uncertainties") if isinstance(parsed.get("uncertainties"), list) else [],
        "confidence": confidence,
    }


def image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _build_prompt(question: str, field_schema: dict[str, Any]) -> str:
    schema_text = json.dumps(field_schema, ensure_ascii=False, indent=2)
    return (
        f"Requested field schema:\n{schema_text}\n\n"
        f"Question: {question}\nKeep field names exactly as provided."
    )


class _OpenAIAnalyzer:
    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("MLLM_API_KEY", "").strip()
        if not api_key or api_key == "replace_me":
            raise RuntimeError("MLLM_API_KEY is missing. Copy .env.example to .env and add your key.")
        self.model = os.getenv("MLLM_MODEL", "gpt-4.1-mini")
        self.api_mode = os.getenv("MLLM_API_MODE", "responses").lower()
        if self.api_mode not in {"responses", "chat"}:
            raise RuntimeError("MLLM_API_MODE must be 'responses' or 'chat'.")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("MLLM_BASE_URL", "https://api.openai.com/v1"),
            timeout=120.0,
            max_retries=2,
        )

    def describe(self):
        return {"backend": "openai", "model": self.model, "api_mode": self.api_mode,
                "system_prompt": SYSTEM_PROMPT}

    def analyze(self, image_path: str | Path, question: str, field_schema: dict[str, Any]):
        prompt = _build_prompt(question, field_schema)
        url = image_data_url(image_path)
        started = time.perf_counter()
        if self.api_mode == "responses":
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": url},
                ]}],
            )
            raw = response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}},
                    ]},
                ],
            )
            raw = response.choices[0].message.content or ""
        result = parse_with_raw(raw, list(field_schema))
        result["raw_output"] = raw
        result["metadata"] = {
            "backend": "openai",
            "model": self.model,
            "api_mode": self.api_mode,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        return result


class _TransformersAnalyzer:
    """Local Qwen2.5-VL inference backend for an NVIDIA GPU."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("The transformers backend requires a CUDA-capable GPU.")

        self.torch = torch
        self.model_name = os.getenv("MLLM_LOCAL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.compute_dtype_name = os.getenv("MLLM_COMPUTE_DTYPE", "bfloat16").strip().lower()
        dtype_options = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if self.compute_dtype_name not in dtype_options:
            raise ValueError("MLLM_COMPUTE_DTYPE must be 'bfloat16' or 'float16'.")
        if self.compute_dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16. Set MLLM_COMPUTE_DTYPE=float16 and validate outputs.")
        compute_dtype = dtype_options[self.compute_dtype_name]
        load_in_4bit = os.getenv("MLLM_LOAD_IN_4BIT", "true").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.min_pixels = int(os.getenv("MLLM_MIN_PIXELS", str(256 * 28 * 28)))
        self.max_pixels = int(os.getenv("MLLM_MAX_PIXELS", str(1024 * 28 * 28)))
        self.max_new_tokens = int(os.getenv("MLLM_MAX_NEW_TOKENS", "512"))
        if not 0 < self.min_pixels <= self.max_pixels or self.max_new_tokens < 1:
            raise ValueError("Invalid pixel or token limits")
        model_kwargs: dict[str, Any] = {
            "torch_dtype": compute_dtype, "device_map": {"": 0}, "attn_implementation": "sdpa",
            "revision": os.getenv("MLLM_MODEL_REVISION", "main"),
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, revision=self.revision or model_kwargs["revision"],
            min_pixels=self.min_pixels, max_pixels=self.max_pixels, use_fast=True)
        self.load_in_4bit = load_in_4bit

    def describe(self):
        return {"backend": "transformers", "model": self.model_name,
                "model_revision": self.revision, "load_in_4bit": self.load_in_4bit,
                "compute_dtype": self.compute_dtype_name, "processor_use_fast": True,
                "attention_implementation": "sdpa", "batch_size": 1,
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                "max_new_tokens": self.max_new_tokens, "do_sample": False,
                "gpu": self.torch.cuda.get_device_name(0), "system_prompt": SYSTEM_PROMPT}

    def analyze(self, image_path: str | Path, question: str, field_schema: dict[str, Any]):
        from qwen_vl_utils import process_vision_info

        prompt = _build_prompt(question, field_schema)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": Path(image_path).resolve().as_uri(),
                 "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                {"type": "text", "text": prompt},
            ]},
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        self.torch.cuda.synchronize(0)
        self.torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        self.torch.cuda.synchronize(0)
        latency = time.perf_counter() - started
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        result = parse_with_raw(raw, list(field_schema))
        result["raw_output"] = raw
        result["metadata"] = {
            "backend": "transformers",
            "model": self.model_name,
            "load_in_4bit": self.load_in_4bit,
            "compute_dtype": self.compute_dtype_name,
            "latency_seconds": round(latency, 3),
            "peak_gpu_memory_gib": round(
                self.torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3
            ),
        }
        return result


class DocumentAnalyzer:
    """Select an API-hosted or local transformers inference backend."""

    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv()
        backend = os.getenv("MLLM_BACKEND", "openai").strip().lower()
        if backend == "openai":
            self._backend = _OpenAIAnalyzer()
        elif backend == "transformers":
            self._backend = _TransformersAnalyzer()
        else:
            raise RuntimeError("MLLM_BACKEND must be 'openai' or 'transformers'.")

    def analyze(self, image_path: str | Path, question: str, field_schema: dict[str, Any]):
        return self._backend.analyze(image_path, question, field_schema)

    def describe(self):
        return self._backend.describe()
