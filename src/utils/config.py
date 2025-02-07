from dataclasses import dataclass


@dataclass
class hp:
    device: str = "auto"  # ["auto", "cuda", "cpu", "mps"]
    model_name: str = "openai/whisper-tiny"
    log_path: str = "src/assets/logging.log"
    test_audio_path: str = "src/assets/test.mp3"
    save_path: str = "src/assets/whisper-tiny-finetuned.pt"
    checkpoint_path: str = "src/assets/whisper-tiny-finetuned.pt"
    random_state: int = 2025

    vocab_path: str = "src/assets/vocab.json"
    max_length: int = 3000

    num_epoches: int = 10
    train_batch_size: int = 8
    test_batch_size: int = 8
    lr: float = 1e-4

    max_speakers_in_audio: int = 5
    min_speakers_in_audio: int = 1

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

    decoder_out_size: int = 384
    proj_size: int = 9

    eos_token_id: int = 1
    speaker_idx_offset: int = 2
    max_target_length: int = 42
