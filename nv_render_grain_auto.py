import os
import time
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

folder_path = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"
output_folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NVoutput"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sample_rate = 22050

# Grain/window settings
# more accurate setting (grain_size = 0.30, hop_size = 0.15)
grain_size_sec = 0.50
hop_size_sec = 0.50

# Similarity threshold
similarity_threshold = 0.80

# Boundary smoothing
fade_length_samples = 256

# Output name
output_file_name = f"auto_grain_render_{int(time.time())}.wav"


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def time_to_samples(time_sec, sr):
    return int(round(time_sec * sr))


def extract_segment(audio, sr, start_sec, end_sec):
    start_sample = max(0, time_to_samples(start_sec, sr))
    end_sample = min(len(audio), time_to_samples(end_sec, sr))
    return audio[start_sample:end_sample]


def extract_segment_samples(audio, start_sample, end_sample):
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    return audio[start_sample:end_sample]


def resample_grain_to_length(grain, target_length):
    if target_length <= 0:
        return np.array([], dtype=np.float32)

    if len(grain) == 0:
        return np.zeros(target_length, dtype=np.float32)

    if len(grain) == target_length:
        return grain.astype(np.float32)

    old_positions = np.linspace(0, 1, len(grain))
    new_positions = np.linspace(0, 1, target_length)
    resized = np.interp(new_positions, old_positions, grain)
    return resized.astype(np.float32)


def apply_edge_crossfade(original_region, replacement_region, fade_length=256):
    if len(original_region) != len(replacement_region):
        raise ValueError("original_region and replacement_region must be the same length")

    output = replacement_region.copy()
    region_length = len(output)

    if region_length == 0:
        return output

    fade_length = min(fade_length, region_length // 2)

    if fade_length <= 0:
        return output

    fade_in = np.linspace(0.0, 1.0, fade_length)
    fade_out = np.linspace(1.0, 0.0, fade_length)

    output[:fade_length] = (
        original_region[:fade_length] * (1.0 - fade_in) +
        replacement_region[:fade_length] * fade_in
    )

    output[-fade_length:] = (
        original_region[-fade_length:] * (1.0 - fade_out) +
        replacement_region[-fade_length:] * fade_out
    )

    return output


def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = (audio / peak) * 0.99
    return audio.astype(np.float32)


def get_feature_vector(audio_segment, sr):
    """
    Simple feature vector:
    MFCC mean + MFCC std
    """
    if len(audio_segment) < 2048:
        return None

    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feature_vector = np.concatenate([mfcc_mean, mfcc_std])
    return feature_vector.reshape(1, -1)


def find_best_match_for_window(n_segment, v_audio_map, sr, grain_size_samples, hop_size_samples):
    """
    Search all V tracks for the best local match to one N segment.
    """
    n_features = get_feature_vector(n_segment, sr)
    if n_features is None:
        return None

    best_score = -1.0
    best_match = None

    for v_track_name, v_audio in v_audio_map.items():
        max_start = len(v_audio) - grain_size_samples
        if max_start <= 0:
            continue

        for v_start in range(0, max_start, hop_size_samples * 4):
            v_end = v_start + grain_size_samples
            v_segment = extract_segment_samples(v_audio, v_start, v_end)

            if len(v_segment) != grain_size_samples:
                continue

            v_features = get_feature_vector(v_segment, sr)
            if v_features is None:
                continue

            score = cosine_similarity(n_features, v_features)[0][0]

            if score > best_score:
                best_score = score
                best_match = {
                    "v_track": v_track_name,
                    "v_start_sample": v_start,
                    "v_end_sample": v_end,
                    "score": float(score)
                }

    return best_match


# ------------------------------------------------------------
# CHOOSE N TRACK
# ------------------------------------------------------------

all_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".mp3")
]

if len(all_files) == 0:
    raise ValueError("No .mp3 files found in the source folder.")

all_files = sorted(all_files)

print("\nAvailable audio files:\n")
for i, file_name in enumerate(all_files):
    print(f"{i}: {file_name}")

while True:
    try:
        choice = int(input("\nSelect the N track by number: "))
        if 0 <= choice < len(all_files):
            n_track_name = all_files[choice]
            break
        else:
            print("Invalid number. Try again.")
    except ValueError:
        print("Please enter a valid number.")

print(f"\nSelected N track: {n_track_name}")

v_track_names = [f for f in all_files if f != n_track_name]

print("\nAvailable V tracks:")
for name in v_track_names:
    print(" ", name)


# ------------------------------------------------------------
# LOAD AUDIO
# ------------------------------------------------------------

print("\nLoading audio files...")

n_path = os.path.join(folder_path, n_track_name)
n_audio, sr = librosa.load(n_path, sr=sample_rate)

v_audio_map = {}
for file_name in v_track_names:
    file_path = os.path.join(folder_path, file_name)
    audio, _ = librosa.load(file_path, sr=sample_rate)
    v_audio_map[file_name] = audio

print(f"N track loaded: {n_track_name}")
print(f"Loaded {len(v_audio_map)} V tracks")


# ------------------------------------------------------------
# PREPARE RENDER
# ------------------------------------------------------------

match_counts = {}
output_audio = n_audio.astype(np.float32).copy()

grain_size_samples = time_to_samples(grain_size_sec, sr)
hop_size_samples = time_to_samples(hop_size_sec, sr)

print("\nSearching for matching grains and replacing where appropriate...\n")

replacement_count = 0
window_count = 0

max_n_start = len(n_audio) - grain_size_samples
for n_start in range(0, max_n_start, hop_size_samples):
    n_end = n_start + grain_size_samples
    n_segment = extract_segment_samples(n_audio, n_start, n_end)

    if len(n_segment) != grain_size_samples:
        continue

    window_count += 1
    print(f"Checking N window {window_count}: {n_start / sr:.2f}s to {n_end / sr:.2f}s")

    best_match = find_best_match_for_window(
        n_segment=n_segment,
        v_audio_map=v_audio_map,
        sr=sr,
        grain_size_samples=grain_size_samples,
        hop_size_samples=hop_size_samples
    )

    if best_match is None:
        print("  No valid match found.")
        continue

    score = best_match["score"]
    print(f"  Best match: {best_match['v_track']} | score = {score:.3f}")

    if score < similarity_threshold:
        print("  Below threshold, keeping original N audio.")
        continue

    v_track = best_match["v_track"]
    v_start = best_match["v_start_sample"]
    v_end = best_match["v_end_sample"]

    match_counts[v_track] = match_counts.get(v_track, 0) + 1

    v_audio = v_audio_map[v_track]
    replacement_grain = extract_segment_samples(v_audio, v_start, v_end)

    if len(replacement_grain) == 0:
        print("  Replacement grain empty, skipping.")
        continue

    replacement_grain = resample_grain_to_length(replacement_grain, grain_size_samples)

    write_end = min(n_start + len(replacement_grain), len(output_audio))
    actual_length = write_end - n_start

    if actual_length <= 0:
        print("  No writable region, skipping.")
        continue

    replacement_grain = replacement_grain[:actual_length]
    original_region = output_audio[n_start:write_end].copy()

    replaced_region = apply_edge_crossfade(
        original_region,
        replacement_grain,
        fade_length=fade_length_samples
    )

    output_audio[n_start:write_end] = replaced_region
    replacement_count += 1

    print("  Replaced.")

print("\nMatch usage by V track:")
for track_name, count in sorted(match_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {track_name}: {count}")

# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------

print(f"\nFinished. Total windows checked: {window_count}")
print(f"Total replacements made: {replacement_count}")

print("\nNormalizing output...")
output_audio = normalize_audio(output_audio)

output_path = os.path.join(output_folder, output_file_name)
sf.write(output_path, output_audio, sr)

print("\nDone.")
print(f"Output written to: {output_path}")