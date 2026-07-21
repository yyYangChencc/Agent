from __future__ import annotations

import re
from threading import Lock


class FlanT5OpinionScorer:
    """使用本地 FLAN-T5 模型把自然语言观念映射到五级量表。"""

    VALID_RATINGS = {-2, -1, 0, 1, 2}

    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None
        self._lock = Lock()

    def score(self, *, topic_statement: str, honest_belief: str) -> int:
        """返回精确的五级观念评分；无效输出直接报错。"""

        self._ensure_loaded()
        prompt = self._build_prompt(topic_statement, honest_belief)
        with self._lock:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
            )
        raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_rating(raw)

    @classmethod
    def parse_rating(cls, raw: str) -> int:
        """只接受独立出现的 -2、-1、0、1、2，拒绝模糊输出。"""

        text = str(raw or "").strip()
        if not re.fullmatch(r"[+-]?[0-2]", text):
            raise ValueError(f"FLAN opinion rating must be one integer in [-2, 2]: {raw!r}")
        rating = int(text)
        if rating not in cls.VALID_RATINGS:
            raise ValueError(f"FLAN opinion rating is out of range: {rating}")
        return rating

    def _ensure_loaded(self) -> None:
        """首次评分时才加载模型，避免其他模式占用显存。"""

        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "FLAN opinion scoring requires torch, transformers and sentencepiece"
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError("FLAN-T5-Large FP16 opinion scoring requires a CUDA GPU")
            self._device = torch.device("cuda")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
            ).to(self._device)
            self._model.eval()

    @staticmethod
    def _build_prompt(topic_statement: str, honest_belief: str) -> str:
        """构造与五级量表方向严格绑定的分类提示词。"""

        return (
            "Classify the person's current honest belief about the exact statement below.\n\n"
            f"Statement:\n{topic_statement}\n\n"
            f"Current honest belief:\n{honest_belief}\n\n"
            "Return exactly one integer and no other text:\n"
            "-2 = strongly opposes the statement\n"
            "-1 = slightly opposes the statement\n"
            "0 = neutral or unclear about the statement\n"
            "1 = slightly supports the statement\n"
            "2 = strongly supports the statement"
        )
