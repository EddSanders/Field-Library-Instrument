import os

# audio library
folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

# gets all mp3 files from foler
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

# choose the N track
choice = int(input("Choose the N track number: "))

if choice < 0 or choice >= len(audio_files):
    print("invalid choice")
    exit()

# set N track
n_track = audio_files[choice]

# all remaining files become V tracks
v_tracks = []
for i, f in enumerate(audio_files):
    if i != choice:
        v_tracks.append(f)

# print results
print("\nNtrack:")
print(n_track)

print("\nVtracks:")
for v in v_tracks:
    print(v)

print("\nNumber of V tracks:", len(v_tracks))
