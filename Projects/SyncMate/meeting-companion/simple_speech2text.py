import torch
import librosa
from transformers import pipeline

# Load model
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
    device=0 if torch.cuda.is_available() else -1
)

# Load audio as numpy array
audio, sr = librosa.load(
    "downloaded_audio.mp3",
    sr=16000,
    mono=True
)

# Pass raw audio instead of filename
result = pipe(
    {"array": audio, "sampling_rate": sr}
)

print(result["text"])