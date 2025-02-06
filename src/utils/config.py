from dataclasses import dataclass


@dataclass
class hp:
    model_name: str = "openai/whisper-tiny"

    max_speakers_in_audio: int = 5
    min_speakers_in_audio: int = 2

    split2path = {
        "train": "src/data/train",
        "test": "src/data/test",
        "valid": "src/data/valid",
    }

    nchannels: int = 1
    sampwidth: int = 2
    framerate: int = 16000
    comptype: str = "NONE"
    compname: str = "not compressed"

    n_mels: int = 80
    n_samples: int = 480000
    padding: int = 0
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_frames: int = 3000
    frames_per_second: int = 100
    # input_stride = 2

    filters_path: str = "src/assets/mel_filters.npz"

    logits_dim: int = 34

    vocab_size: int = 51865
    proj_size: int = max_speakers_in_audio + 3
