from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class RAGGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None, max_new_tokens: int = 128, temperature: float = 0.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _format_context(self, contexts: List[str], max_tokens: int = 4000) -> str:
        joined = "\n\n".join(contexts)
        tokens = self.tokenizer(joined, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    def build_prompt(self, question: str, contexts: List[str]) -> List[dict]:

        # Chat template messages; transformers will apply chat formatting for Qwen
        system_msg = "You are a helpful assistant. Answer strictly using the provided context in few words."
        context_block = self._format_context(contexts)
        user_msg = f"Question:\n{question}\n\nContext:\n{context_block}"
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def generate(self, question: str, contexts: List[str]) -> str:
        messages = self.build_prompt(question, contexts)
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                top_k=50 if self.temperature > 0.0 else None,
                top_p=0.9 if self.temperature > 0.0 else None,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return text.strip()
