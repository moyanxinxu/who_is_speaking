from dataclasses import dataclass


@dataclass
class hp:
    max_speakers_in_audio = 5
    min_speakers_in_audio = 2

    split2path = {
        "train": "src/data/train",
        "test": "src/data/test",
        "valid": "src/data/valid",
    }

    nchannels = 1
    sampwidth = 2
    framerate = 16000
    comptype = "NONE"
    compname = "not compressed"

    n_mels = 80
    n_samples = 480000
    padding = 0
    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    n_frames = 3000
    frames_per_second = 100
    # input_stride = 2

    filters_path = "src/assets/mel_filters.npz"

    logits_dim = 34
