"""Load a trained checkpoint and sample from the custom model.

The student model is a custom architecture rather than a HuggingFace model, so
it cannot be served by vLLM. This wrapper provides plain continuation and
chat-style response helpers used by the evaluator.
"""

from .model import GPT, build_config
from .sftstage import verify_checkpoint_tokenizer
from .tokenizer import SyntheticTokenizer
from .utils import normalize_state_dict


class StudentModel:
    """A trained checkpoint with plain continuation and instruction helpers.

    tokenizer_path defaults to the config's own tokenizer; a stage whose
    checkpoints live in a different tree from the tokenizer that built them
    (the simulator, which trains into its own run but tokenizes with the
    base run's artifact) passes that artifact explicitly.
    """

    def __init__(self, config, checkpoint_path, device=None,
                 tokenizer_path=None):
        import torch

        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        saved = torch.load(checkpoint_path, map_location=self.device)
        if tokenizer_path is None:
            tokenizer_path = config.tokenizer_path
        verify_checkpoint_tokenizer(saved, checkpoint_path, tokenizer_path)
        self.tokenizer = SyntheticTokenizer(tokenizer_path)
        gpt_config = build_config(config.model, saved['vocabulary_size'])
        self.model = GPT(gpt_config).to(self.device).eval()
        self.model.load_state_dict(normalize_state_dict(saved['model']))
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
        """Answer an instruction using the light Question and Answer framing."""
        import torch

        from .data import INSTRUCTION_PREFIX

        prompt_text = INSTRUCTION_PREFIX % instruction.strip()
        token_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt_text)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        output = self.model.generate(
            input_ids, max_new_tokens, temperature=temperature, top_p=top_p,
            eos_id=self.tokenizer.eos_id, repetition_penalty=repetition_penalty,
        )
        generated = output[0, len(token_ids):].tolist()
        return self.tokenizer.decode(generated).strip()
