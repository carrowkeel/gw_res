"""Stage one: train a fresh BPE tokenizer on the synthetic corpus only.

Training the tokenizer from scratch is the main safeguard against leakage: the
vocabulary can only contain subwords that appear in the referent-free corpus,
so the model has no tokens for unseen real-world entities.

Digits are pre-tokenized individually, so no multi-digit token can ever form.
A byte-level BPE trained on free text otherwise learns frequency-merged number
chunks (' 15', ' 193', ' 24') that vary per number and destroy the place-value
alignment arithmetic depends on, which was diagnosed as the ceiling on the
arithmetic SFT. The build verifies the property and refuses to save a tokenizer
that violates it.

    python -m slm.tokenizer --config configs/poc.yaml
    python -m slm.tokenizer --check runs/<tree>/tokenizer/tokenizer.json
"""

import argparse
import hashlib
import json
import re

from .config import load_config
from .utils import ensure_directory, get_logger

logger = get_logger('tokenizer')

_MULTI_DIGIT = re.compile(r'\d\d')


def multi_digit_tokens(tokenizer):
    """Return vocabulary tokens that span more than one digit.

    Digit runs are split into single-digit pre-tokens before BPE, and BPE
    never merges across pre-token boundaries, so a well-built vocabulary
    has none of these. Any that appear mean the digit pre-tokenizer is
    missing and place-value alignment is broken.
    """
    return sorted(
        token for token in tokenizer.get_vocab()
        if _MULTI_DIGIT.search(token)
    )


def fingerprint(path):
    """Return a short digest identifying a saved tokenizer artifact.

    Packed data and checkpoints record the fingerprint of the tokenizer
    they were built with, so a stage can refuse artifacts from a different
    tokenizer instead of silently training on a mismatched vocabulary.
    """
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def iterate_corpus_texts(config):
    """Yield every pretraining and finetuning text used to train the tokenizer."""
    pretrain_directory = config.corpus_pretrain_dir
    for shard in sorted(pretrain_directory.glob('shard_*.jsonl')):
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)['text']
    pairs_path = config.corpus_sft_path
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
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
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

    offenders = multi_digit_tokens(tokenizer)
    if offenders:
        raise ValueError(
            'tokenizer learned %d multi-digit tokens (e.g. %s); the digit '
            'pre-tokenizer is not in effect and place-value alignment is '
            'broken' % (len(offenders), offenders[:10])
        )
    probe = 'the price rose 15 points to 1545 in q3.'
    decoded = tokenizer.decode(tokenizer.encode(probe).ids)
    if decoded != probe:
        raise ValueError(
            'tokenizer does not round-trip text with numbers: %r became %r'
            % (probe, decoded)
        )

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


def check_digit_tokenization(path):
    """Report how a saved tokenizer splits numbers; exit non-zero on merges."""
    import random

    tokenizer = SyntheticTokenizer(path)
    random_generator = random.Random(0)
    numbers = (
        list(range(0, 10))
        + [random_generator.randint(10, 99) for _ in range(6)]
        + [random_generator.randint(100, 999) for _ in range(6)]
        + [random_generator.randint(1000, 1999) for _ in range(4)]
    )
    print('%-8s %-18s %s' % ('number', 'token_ids', 'pieces'))
    for number in numbers:
        ids = tokenizer.encode(str(number))
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        print('%-8s %-18s %r' % (number, ids, pieces))
    offenders = multi_digit_tokens(tokenizer.tokenizer)
    if offenders:
        print('\nFAIL: %d multi-digit tokens in vocabulary, e.g. %s'
              % (len(offenders), offenders[:10]))
        raise SystemExit(1)
    probe = 'the price rose 15 points to 1545 in q3.'
    decoded = tokenizer.decode(tokenizer.encode(probe))
    if decoded != probe:
        print('\nFAIL: numbers do not round-trip: %r became %r'
              % (probe, decoded))
        raise SystemExit(1)
    print('\nOK: every numeric token is a single digit and numbers round-trip')


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--config')
    parser.add_argument(
        '--check',
        help='path to a saved tokenizer to report digit tokenization for, '
             'instead of training',
    )
    arguments = parser.parse_args()
    if arguments.check:
        check_digit_tokenization(arguments.check)
        return
    if not arguments.config:
        raise SystemExit('either --config (to train) or --check is required')
    train(load_config(arguments.config))


if __name__ == '__main__':
    main()
