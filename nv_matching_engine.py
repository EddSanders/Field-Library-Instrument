import os
import librosa
import numpy as np

# -----------------------------------
# SETTINGS
# -----------------------------------

folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

sample_rate = 22050
start_window_seconds = 0.5   # starting comparison window
step_seconds = 0.25          # how far to move along N each time
search_step_seconds = 0.25   # how far to move through V tracks
similarity_threshold = 0.90  # higher = stricter match

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

def get_feature_vector(audio_section, sr):
    # turn audio into MFCC features, then average them
    mfcc = librosa.feature.mfcc(y=audio_section, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def cosine_similarity(vec1, vec2):
    # compare two feature vectors
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)

# -----------------------------------
# GET FILES
# -----------------------------------

files = os.listdir(folder)

audio_files = []
for f in files:
    if f.endswith(".mp3"):
        audio_files.append(f)

audio_files.sort()

print("Available files:")
for i, f in enumerate(audio_files):
    print(i, f)

choice = int(input("Choose the N track number: "))

if choice < 0 or choice >= len(audio_files):
    print("Invalid choice")
    exit()

n_track = audio_files[choice]

v_tracks = []
for i, f in enumerate(audio_files):
    if i != choice:
        v_tracks.append(f)

print("\nN track:", n_track)
print("Number of V tracks:", len(v_tracks))

# -----------------------------------
# LOAD N TRACK
# -----------------------------------

n_path = os.path.join(folder, n_track)
n_audio, n_sr = librosa.load(n_path, sr=sample_rate)

window_samples = int(start_window_seconds * sample_rate)
step_samples = int(step_seconds * sample_rate)
search_step_samples = int(search_step_seconds * sample_rate)

# -----------------------------------
# MATCH ENGINE
# -----------------------------------

print("\nBest matches:\n")

n_start = 0

while n_start + window_samples <= len(n_audio):
    n_end = n_start + window_samples
    n_section = n_audio[n_start:n_end]
    n_features = get_feature_vector(n_section, n_sr)

    best_score = -1
    best_v_track = None
    best_v_start = None
    best_v_end = None

    # compare this N section against all V tracks
    for v_name in v_tracks:
        v_path = os.path.join(folder, v_name)
        v_audio, v_sr = librosa.load(v_path, sr=sample_rate)

        v_start = 0
        while v_start + window_samples <= len(v_audio):
            v_end = v_start + window_samples
            v_section = v_audio[v_start:v_end]
            v_features = get_feature_vector(v_section, v_sr)

            score = cosine_similarity(n_features, v_features)

            if score > best_score:
                best_score = score
                best_v_track = v_name
                best_v_start = v_start
                best_v_end = v_end

            v_start += search_step_samples

    print("N section:",
          round(n_start / sample_rate, 2), "to",
          round(n_end / sample_rate, 2), "seconds")

    print("Best V match:", best_v_track)
    print("V section:",
          round(best_v_start / sample_rate, 2), "to",
          round(best_v_end / sample_rate, 2), "seconds")

    print("Similarity score:", round(best_score, 4))

    if best_score >= similarity_threshold:
        print("Match accepted\n")
    else:
        print("Match weak\n")

    n_start += step_samples