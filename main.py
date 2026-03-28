import librosa
import sounddevice as sd

audio, sr = librosa.load("test.wav")

print("Loaded audio")