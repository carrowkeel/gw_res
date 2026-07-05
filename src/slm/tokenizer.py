"""Stage one: train a fresh BPE tokenizer on the synthetic corpus only.

Training the tokenizer from scratch is the main safeguard against leakage: the
vocabulary can only contain subwords that appear in the referent-free corpus,
so the model has no tokens for unseen real-world entities.

    python -m slm.tokenizer --config configs/poc.yaml
"""

import argparse
import json

from .config import load_config
from .utils import ensure_directory, get_logger

logger = get_logger('tokenizer')


def iterate_corpus_texts(config):
    """Yield every pretraining and finetuning text used to train the tokenizer."""
    pretrain_directory = config.data_dir / 'pretrain'
    for shard in sorted(pretrain_directory.glob('shard_*.jsonl')):
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)['text']
    pairs_path = config.data_dir / 'sft' / 'sft.jsonl'
    if pairs_path.exists():
        from .data import render_instruction

        with open(pairs_path) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    record = json.loads(stripped)
                    yield render_instruction(record['prompt'], record['response'])


def train(config, extra_special_tokens=None, output_path=None):
    """Train and save a byte-level BPE tokenizer on the synthetic corpus.

    extra_special_tokens and output_path let a variant pipeline train its own
    tokenizer with additional reserved tokens at a different location without
    touching the default artifact.
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC

    tokenizer_config = config.tokenizer
    tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = list(tokenizer_config.special_tokens)
    for token in (extra_special_tokens or []):
        if token not in special_tokens:
            special_tokens.append(token)
    if '<|unk|>' not in special_tokens:
        special_tokens = ['<|unk|>'] + special_tokens

    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config.vocabulary_size,
        min_frequency=tokenizer_config.minimum_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    logger.info(
        'training BPE (vocab size %d) on synthetic corpus',
        tokenizer_config.vocabulary_size,
    )
    tokenizer.train_from_iterator(iterate_corpus_texts(config), trainer=trainer)

    if output_path is None:
        output_path = config.tokenizer_path
    ensure_directory(output_path.parent)
    tokenizer.save(str(output_path))
    logger.info(
        'saved tokenizer to %s (vocab %d)',
        output_path, tokenizer.get_vocab_size(),
    )
    return output_path


class SyntheticTokenizer:
    """Runtime wrapper exposing the operations the pipeline needs."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocabulary_size = self.tokenizer.get_vocab_size()
        self.bos_id = self._token_id('<|bos|>')
        self.eos_id = self._token_id('<|eos|>')
        self.pad_id = self._token_id('<|pad|>')
        self.user_id = self._token_id('<|user|>')
        self.assistant_id = self._token_id('<|assistant|>')

    def _token_id(self, token):
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError('special token %r missing from tokenizer' % token)
        return token_id

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    train(load_config(arguments.config))


if __name__ == '__main__':
    main()
