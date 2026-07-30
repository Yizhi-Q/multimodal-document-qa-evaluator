"""Provider configuration, multimodal request construction, and JSON parsing."""

from __future__ import annotations

import base64
import json
import mimetypes
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

    fields = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else {}
    # Some vision-language models follow the requested field schema directly
    # and return the extracted fields at the top level. Accept that useful
    # output instead of incorrectly scoring every requested field as missing.
    if not fields and expected_fields:
        fields = {name: parsed.get(name) for name in expected_fields}
    if expected_fields:
        fields = {name: fields.get(name) for name in expected_fields}
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence"))))
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
        )

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
        result = parse_model_output(raw, list(field_schema))
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
        load_in_4bit = os.getenv("MLLM_LOAD_IN_4BIT", "true").strip().lower() in {
            "1", "true", "yes", "on",
        }
        model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.load_in_4bit = load_in_4bit

    def analyze(self, image_path: str | Path, question: str, field_schema: dict[str, Any]):
        from qwen_vl_utils import process_vision_info

        prompt = _build_prompt(question, field_schema)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": Path(image_path).resolve().as_uri()},
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

        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(os.getenv("MLLM_MAX_NEW_TOKENS", "512")),
                do_sample=False,
            )
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
        result = parse_model_output(raw, list(field_schema))
        result["metadata"] = {
            "backend": "transformers",
            "model": self.model_name,
            "load_in_4bit": self.load_in_4bit,
            "latency_seconds": round(latency, 3),
            "peak_gpu_memory_gib": round(
                self.torch.cuda.max_memory_allocated() / (1024 ** 3), 3
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
