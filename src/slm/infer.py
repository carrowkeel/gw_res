"""Load a trained checkpoint and sample from the custom model.

The student model is a custom architecture rather than a HuggingFace model, so
it cannot be served by vLLM. This wrapper provides plain continuation and
chat-style response helpers used by the evaluator.
"""

from .model import GPT, build_config
from .tokenizer import SyntheticTokenizer


class StudentModel:
    def __init__(self, config, checkpoint_path, device=None):
        import torch

        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        saved = torch.load(checkpoint_path, map_location=self.device)
        self.tokenizer = SyntheticTokenizer(config.tokenizer_path)
        gpt_config = build_config(config.model, saved['vocabulary_size'])
        self.model = GPT(gpt_config).to(self.device).eval()
        self.model.load_state_dict(saved['model'])
        self.block_size = gpt_config.block_size

    def complete(self, text, max_new_tokens=256, temperature=0.8, top_p=0.95,
                 repetition_penalty=1.0):
        """Continue raw text in the pretraining style."""
        import torch

        token_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(text)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        output = self.model.generate(
            input_ids, max_new_tokens, temperature=temperature, top_p=top_p,
            eos_id=self.tokenizer.eos_id, repetition_penalty=repetition_penalty,
        )
        generated = output[0, len(token_ids):].tolist()
        return self.tokenizer.decode(generated)

    def respond(self, instruction, max_new_tokens=256, temperature=0.8,
                top_p=0.95, repetition_penalty=1.0):
        """Produce a chat-style response using the finetuning framing."""
        import torch

        token_ids = (
            [self.tokenizer.bos_id, self.tokenizer.user_id]
            + self.tokenizer.encode(instruction)
            + [self.tokenizer.assistant_id]
        )
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        output = self.model.generate(
            input_ids, max_new_tokens, temperature=temperature, top_p=top_p,
            eos_id=self.tokenizer.eos_id, repetition_penalty=repetition_penalty,
        )
        generated = output[0, len(token_ids):].tolist()
        return self.tokenizer.decode(generated)
