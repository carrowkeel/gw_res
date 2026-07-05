"""Configuration schema for the pipeline.

The whole pipeline is driven by a single YAML file. Every stage reads the same
configuration object so parameters live in one place and the Slurm layer can
inspect resource requests without importing heavy dependencies.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class ProjectConfig:
    name: str = 'slm-poc'
    out_dir: str = 'runs/poc'
    seed: int = 1337
    corpus_dir: str = None


@dataclass
class GenerateConfig:
    default_model: str = 'Qwen/Qwen2.5-7B-Instruct'
    type_models: dict = field(default_factory=dict)
    severity: str = 's1'
    text_type_weights: dict = field(
        default_factory=lambda: {
            'prose': 1.0,
            'conversation': 1.0,
            'definition': 1.0,
            'description': 1.0,
            'reasoning': 1.0,
        }
    )
    number_of_texts: int = 100000
    number_of_pairs: int = 20000
    temperature: float = 1.0
    top_p: float = 0.95
    frequency_penalty: float = 0.4
    presence_penalty: float = 0.0
    max_tokens: int = 512
    workers: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048
    dtype: str = 'bfloat16'
    batch_size: int = 1024
    apply_filter: bool = True
    deduplicate: bool = True
    minimum_characters: int = 200


@dataclass
class TokenizerConfig:
    vocabulary_size: int = 16000
    minimum_frequency: int = 2
    special_tokens: list = field(
        default_factory=lambda: [
            '<|pad|>',
            '<|bos|>',
            '<|eos|>',
            '<|user|>',
            '<|assistant|>',
            '<|endoftext|>',
        ]
    )


@dataclass
class ModelConfig:
    preset: str = 'poc-150m'
    number_of_layers: int = None
    number_of_heads: int = None
    number_of_key_value_heads: int = None
    embedding_dimension: int = None
    block_size: int = None
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    vocabulary_size: int = None


@dataclass
class PretrainConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    maximum_steps: int = 50000
    learning_rate: float = 6e-4
    minimum_learning_rate: float = 6e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 1.0
    dtype: str = 'bfloat16'
    compile_model: bool = True
    log_interval: int = 20
    evaluation_interval: int = 1000
    evaluation_iterations: int = 100
    checkpoint_interval: int = 1000
    validation_fraction: float = 0.005
    early_stop_patience: int = 0
    instruction_fraction: float = 0.1


@dataclass
class FinetuneConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    maximum_steps: int = None
    learning_rate: float = 2e-5
    minimum_learning_rate: float = 2e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    dtype: str = 'bfloat16'
    compile_model: bool = False
    log_interval: int = 10
    checkpoint_interval: int = 500
    maximum_sequence_length: int = 1024


@dataclass
class EvalConfig:
    judge_model: str = None
    number_of_generation_samples: int = 200
    number_of_probe_questions: int = 100
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    student_gpu_memory_utilization: float = 0.45
    judge_gpu_memory_utilization: float = 0.45


@dataclass
class GraphConfig:
    segment_tokens: int = 48
    node_token_limit: int = 200
    relatedness_threshold: float = 0.1
    examples_per_text: int = 6
    context_dropout: float = 0.1
    holdout_fraction: float = 0.02
    export_intent_examples: int = 3
    context_budgets: list = field(default_factory=lambda: [64, 128, 256])
    number_of_eval_conversations: int = 50
    max_new_tokens: int = 96
    judge_enabled: bool = True


@dataclass
class ScaleConfig:
    rungs: list = field(default_factory=list)


@dataclass
class SlurmConfig:
    enabled: bool = True
    partition: str = None
    account: str = None
    gres: str = 'gpu:l40s:1'
    memory: str = '64G'
    cpus_per_task: int = 8
    time_limit: str = '24:00:00'
    pretrain_gres: str = None
    log_dir: str = 'runs/poc/slurm_logs'
    extra_sbatch: list = field(default_factory=list)
    cache_dir: str = None
    environment: dict = field(default_factory=dict)


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    @property
    def out_dir(self):
        return Path(self.project.out_dir)

    @property
    def data_dir(self):
        return self.out_dir / 'data'

    @property
    def corpus_pretrain_dir(self):
        base = Path(self.project.corpus_dir) if self.project.corpus_dir else self.data_dir
        return base / 'pretrain'

    @property
    def corpus_sft_path(self):
        base = Path(self.project.corpus_dir) if self.project.corpus_dir else self.data_dir
        return base / 'sft' / 'sft.jsonl'

    @property
    def tokenizer_path(self):
        return self.out_dir / 'tokenizer' / 'tokenizer.json'

    @property
    def pretrain_dir(self):
        return self.out_dir / 'checkpoints' / 'pretrain'

    @property
    def sft_dir(self):
        return self.out_dir / 'checkpoints' / 'sft'

    @property
    def eval_dir(self):
        return self.out_dir / 'eval'

    @property
    def graphs_dir(self):
        return self.data_dir / 'graphs'

    @property
    def graph_packed_dir(self):
        return self.data_dir / 'graph_packed'

    @property
    def graph_tokenizer_path(self):
        return self.out_dir / 'tokenizer' / 'graph_tokenizer.json'

    @property
    def graph_pretrain_dir(self):
        return self.out_dir / 'checkpoints' / 'graph_pretrain'


_SECTION_TYPES = {
    'project': ProjectConfig,
    'generate': GenerateConfig,
    'tokenizer': TokenizerConfig,
    'model': ModelConfig,
    'pretrain': PretrainConfig,
    'finetune': FinetuneConfig,
    'eval': EvalConfig,
    'graph': GraphConfig,
    'scale': ScaleConfig,
    'slurm': SlurmConfig,
}


def _build_section(section_type, data):
    known_names = {single_field.name for single_field in fields(section_type)}
    arguments = {}
    for key, value in (data or {}).items():
        if key not in known_names:
            raise ValueError(
                'Unknown config key %r for %s' % (key, section_type.__name__)
            )
        arguments[key] = value
    return section_type(**arguments)


def load_config(path):
    """Load a Config from a YAML file, validating section keys."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError('Unknown config sections: %s' % sorted(unknown))
    sections = {}
    for name, section_type in _SECTION_TYPES.items():
        sections[name] = _build_section(section_type, raw.get(name))
    return Config(**sections)


def to_dict(value):
    """Recursively convert a dataclass tree into plain dictionaries."""
    if is_dataclass(value):
        return {
            single_field.name: to_dict(getattr(value, single_field.name))
            for single_field in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value


def save_config(config, path):
    """Write the resolved configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False)
