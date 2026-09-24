"""Global configuration objects, loaded from YAML files (see `configs/`)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

STEMS = ("vocals", "drums", "bass", "other")
N_STEMS = len(STEMS)


@dataclass
class STFTConfig:
    """Short-Time Fourier Transform used to turn audio into an image."""

    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None
    center: bool = True
    window: str = "hann"

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def seconds_per_frame(self) -> float:
        return self.hop_length


@dataclass
class ModelConfig:
    """U-Net segmentation model."""

    base_channels: int = 32
    depth: int = 5
    norm: str = "batch"          # batch | group | instance
    up_mode: str = "bilinear"    # bilinear | transpose
    mask: str = "tanh"           # tanh -> complex ratio mask (cIRM)
    dropout: float = 0.0


@dataclass
class AugmentConfig:
    """Data augmentation, essential because MUSDB18 train has only ~100 songs."""

    gain_db: float = 6.0             # per-source random gain
    channel_swap_p: float = 0.5      # independently swap L/R per source
    polarity_p: float = 0.5          # randomly invert polarity per source
    stem_swap_p: float = 0.5         # replace a stem with one from another song
    drop_source_p: float = 0.05      # silence one source entirely
    remix_p: float = 0.3             # replace *all* stems with another song's stems


@dataclass
class DataConfig:
    root: str = "data/musdb18"
    sample_rate: int = 44100
    channels: int = 2
    chunk_seconds: float = 6.0
    segment_seconds: float = 45.0    # decoded slice of a song per visit (0 = whole song)
    chunks_per_load: int = 6         # crops taken from each decoded slice
    valid_tracks: int = 14           # first N sorted train tracks held out for validation
    cache_tracks: int = 8            # decoded slices kept in RAM per dataloader worker
    num_workers: int = 4
    aug: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class LossConfig:
    waveform: float = 1.0            # L1 on the reconstructed waveform
    magnitude: float = 0.5           # L1 on compressed magnitudes
    magnitude_compress: float = 0.3
    si_sdr: float = 0.0              # -SI-SDR/10 term (scale-invariant, dB-driven)
    consistency: bool = True         # project estimates so they sum back to the mix


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 120
    steps_per_epoch: int = 400
    warmup_steps: int = 500
    grad_clip: float = 5.0
    amp: bool = False
    ema_decay: float = 0.999
    max_hours: float | None = None   # stop cleanly after this much wall-clock time
    seed: int = 1234
    device: str = "auto"             # auto | mps | cuda | cpu
    out_dir: str = "runs/unet"
    ckpt_dir: str = "checkpoints"
    val_chunks: int = 12
    val_every: int = 400             # steps
    log_every: int = 25


@dataclass
class Config:
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS: dict[str, type] = {
    "stft": STFTConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "loss": LossConfig,
    "train": TrainConfig,
}


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate `cls` from a dict, ignoring unknown keys (forward compatible)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config (optional) and apply dotted `section.key` overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config()
    for section, cls in _SECTIONS.items():
        values = dict(raw.get(section) or {})
        if section == "data" and "aug" in values:
            cfg.data.aug = _dataclass_from_dict(AugmentConfig, dict(values.pop("aug") or {}))
        setattr(cfg, section, _dataclass_from_dict(cls, values))

    for key, value in overrides.items():
        if value is None:
            continue
        section, _, name = key.partition(".")
        if not name:
            raise KeyError(f"override '{key}' must look like 'train.lr'")
        target = getattr(cfg, section)
        if not hasattr(target, name):
            raise KeyError(f"unknown config field '{section}.{name}'")
        setattr(target, name, value)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))


def config_json(cfg: Config) -> str:
    return json.dumps(asdict(cfg), indent=2, sort_keys=False)
