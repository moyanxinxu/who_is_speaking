from src.modules import WhisperForDiarization
from src.utils import hp

model = WhisperForDiarization()

model.load_params(hp.save_path)

output = model.predict(hp.test_audio_path)

print(output)
