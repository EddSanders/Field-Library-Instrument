import os
import librosa

# folder containing one family of similar sounds
folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

# get all mp3 files
files = os.listdir(folder)

audio_files = []
for f in files:
    if f.endswith(".mp3"):
        audio_files.append(f)

audio_files.sort()

# show files with numbers
print("Available files:")
for i, f in enumerate(audio_files):
    print(i, f)

# choose N track
choice = int(input("Choose the N track number: "))

if choice < 0 or choice >= len(audio_files):
    print("Invalid choice")
    exit()

# define N and V tracks
n_track = audio_files[choice]

v_tracks = []
for i, f in enumerate(audio_files):
    if i != choice:
        v_tracks.append(f)

# load N track
n_path = os.path.join(folder, n_track)
n_audio, n_sr = librosa.load(n_path)

print("\nN track loaded:")
print("Name:", n_track)
print("Sample rate:", n_sr)
print("Duration:", len(n_audio) / n_sr)
print("Max amplitude:", max(n_audio))

# load V tracks
print("\nV tracks loaded:")
for v in v_tracks:
    v_path = os.path.join(folder, v)
    v_audio, v_sr = librosa.load(v_path)

    print("\nName:", v)
    print("Sample rate:", v_sr)
    print("Duration:", len(v_audio) / v_sr)
    print("Max amplitude:", max(v_audio))