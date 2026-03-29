import os
import librosa
import numpy as np

folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

files = os.listdir(folder)

audio_files = []
for f in files:
    if f.endswith(".mp3"):
        audio_files.append(f)

audio_files.sort()

# show files
print("Available files:")
for i, f in enumerate(audio_files):
    print(i, f)

# choose N track
choice = int(input("Choose the N track number: "))

if choice < 0 or choice >= len(audio_files):
    print("Invalid choice")
    exit()

n_track = audio_files[choice]

v_tracks = []
for i, f in enumerate(audio_files):
    if i != choice:
        v_tracks.append(f)

# --- LOAD N TRACK ---
n_path = os.path.join(folder, n_track)
n_audio, n_sr = librosa.load(n_path)

n_duration = len(n_audio) / n_sr
n_max_amp = max(n_audio)
n_mean_amp = np.mean(np.abs(n_audio))

print("\nN track:", n_track)

# --- COMPARE WITH V TRACKS ---
print("\nComparison results:\n")

for v in v_tracks:
    v_path = os.path.join(folder, v)
    v_audio, v_sr = librosa.load(v_path)

    v_duration = len(v_audio) / v_sr
    v_max_amp = max(v_audio)
    v_mean_amp = np.mean(np.abs(v_audio))

    # simple differences
    duration_diff = abs(n_duration - v_duration)
    max_amp_diff = abs(n_max_amp - v_max_amp)
    mean_amp_diff = abs(n_mean_amp - v_mean_amp)

    # simple score (lower = more similar)
    score = duration_diff + max_amp_diff + mean_amp_diff

    print("V track:", v)
    print("  Duration diff:", duration_diff)
    print("  Max amp diff:", max_amp_diff)
    print("  Mean amp diff:", mean_amp_diff)
    print("  Similarity score:", score)
    print()