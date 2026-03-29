import os
import librosa
import numpy as np

# -----------------------------------
# SETTINGS
# -----------------------------------

folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

sample_rate = 22050

start_window_seconds = 0.5     # initial test window
grow_step_seconds = 0.25       # how much the match grows each time
n_step_seconds = 0.5           # how far to move along N after each accepted region

similarity_threshold = 0.98    # minimum similarity to keep growing
min_region_seconds = 0.5       # shortest accepted region

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

def get_feature_vector(audio_section, sr):
    # turn audio into MFCC features, then average them into one vector
    mfcc = librosa.feature.mfcc(y=audio_section, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def cosine_similarity(vec1, vec2):
    # compare two feature vectors
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

def find_best_match_for_window(n_audio, n_start, window_samples, v_tracks, folder, sr):
    # find the best V match for one N window
    n_end = n_start + window_samples
    n_section = n_audio[n_start:n_end]
    n_features = get_feature_vector(n_section, sr)

    best_score = -1
    best_v_track = None
    best_v_start = None

    for v_name in v_tracks:
        v_path = os.path.join(folder, v_name)
        v_audio, v_sr = librosa.load(v_path, sr=sr)

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

            v_start += window_samples

    return best_v_track, best_v_start, best_score

def grow_match(n_audio, n_start, v_audio, v_start, sr, start_window_samples, grow_step_samples, threshold):
    # start with a small match, then grow it while similarity stays high
    current_size = start_window_samples
    last_good_size = 0
    last_good_score = -1

    while True:
        n_end = n_start + current_size
        v_end = v_start + current_size

        if n_end > len(n_audio) or v_end > len(v_audio):
            break

        n_section = n_audio[n_start:n_end]
        v_section = v_audio[v_start:v_end]

        n_features = get_feature_vector(n_section, sr)
        v_features = get_feature_vector(v_section, sr)

        score = cosine_similarity(n_features, v_features)

        if score >= threshold:
            last_good_size = current_size
            last_good_score = score
            current_size += grow_step_samples
        else:
            break

    return last_good_size, last_good_score

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

start_window_samples = int(start_window_seconds * sample_rate)
grow_step_samples = int(grow_step_seconds * sample_rate)
n_step_samples = int(n_step_seconds * sample_rate)
min_region_samples = int(min_region_seconds * sample_rate)

# -----------------------------------
# ADAPTIVE MATCH ENGINE
# -----------------------------------

matches = []
n_start = 0

print("\nAdaptive matches:\n")

while n_start + start_window_samples <= len(n_audio):
    # first: find best V match for the starting window
    best_v_track, best_v_start, best_score = find_best_match_for_window(
        n_audio, n_start, start_window_samples, v_tracks, folder, sample_rate
    )

    if best_v_track is None:
        n_start += n_step_samples
        continue

    # load the chosen V track
    best_v_path = os.path.join(folder, best_v_track)
    v_audio, v_sr = librosa.load(best_v_path, sr=sample_rate)

    # then: try to grow the region
    region_size, final_score = grow_match(
        n_audio,
        n_start,
        v_audio,
        best_v_start,
        sample_rate,
        start_window_samples,
        grow_step_samples,
        similarity_threshold
    )

    if region_size >= min_region_samples:
        n_end = n_start + region_size
        v_end = best_v_start + region_size

        match_info = {
            "n_start_sec": n_start / sample_rate,
            "n_end_sec": n_end / sample_rate,
            "v_track": best_v_track,
            "v_start_sec": best_v_start / sample_rate,
            "v_end_sec": v_end / sample_rate,
            "score": final_score
        }

        matches.append(match_info)

        print("N region:",
              round(match_info["n_start_sec"], 2), "to",
              round(match_info["n_end_sec"], 2), "seconds")

        print("Matched V track:", match_info["v_track"])
        print("V region:",
              round(match_info["v_start_sec"], 2), "to",
              round(match_info["v_end_sec"], 2), "seconds")

        print("Final similarity score:", round(match_info["score"], 4))
        print()

        # jump forward to end of accepted region
        n_start = n_end
    else:
        # no good region found, move forward a little
        n_start += n_step_samples

print("Total matches found:", len(matches))