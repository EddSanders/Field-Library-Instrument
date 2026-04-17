import os
import time
import librosa
import numpy as np
import soundfile as sf


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

# Source library folder
folder_path = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

# Output folder for rendered files
output_folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NVoutput"

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Sample rate used for loading all audio
sample_rate = 22050

# Number of samples used for crossfade at grain boundaries
fade_length_samples = 256

# Output file name
output_file_name = f"grain_render_{int(time.time())}.wav"


# ------------------------------------------------------------
# MATCHES
# Replace these with real matches from your matching stage
# The N track is chosen interactively when the script runs
# The V tracks must be OTHER files from the source folder
# ------------------------------------------------------------

matches = [
    {
        "n_start_sec": 0.50,
        "n_end_sec": 1.10,
        "v_track": "PelophylaxLessonae01.mp3",
        "v_start_sec": 0.20,
        "v_end_sec": 0.80,
        "score": 0.91
    },
    {
        "n_start_sec": 1.30,
        "n_end_sec": 1.90,
        "v_track": "PelophylaxLessonae02.mp3",
        "v_start_sec": 0.40,
        "v_end_sec": 1.00,
        "score": 0.88
    },
    {
        "n_start_sec": 2.10,
        "n_end_sec": 2.80,
        "v_track": "PelophylaxLessonae03.mp3",
        "v_start_sec": 0.10,
        "v_end_sec": 0.90,
        "score": 0.86
    }
]


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def time_to_samples(time_sec, sr):
    return int(round(time_sec * sr))


def extract_segment(audio, sr, start_sec, end_sec):
    start_sample = max(0, time_to_samples(start_sec, sr))
    end_sample = min(len(audio), time_to_samples(end_sec, sr))
    return audio[start_sample:end_sample]


def resample_grain_to_length(grain, target_length):
    """
    Resize a grain to fit the target N-region length.
    Uses simple interpolation to keep the code easy to understand.
    """
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
    """
    Replaces a region, but crossfades the edges so transitions are smoother.
    The middle is mostly replacement audio, while the edges blend with N.
    """
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

    # Blend start
    output[:fade_length] = (
        original_region[:fade_length] * (1.0 - fade_in) +
        replacement_region[:fade_length] * fade_in
    )

    # Blend end
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


# ------------------------------------------------------------
# CHOOSE N TRACK INTERACTIVELY
# ------------------------------------------------------------

# Only use mp3 source files from the library
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

# Everything else becomes V
v_track_names = [f for f in all_files if f != n_track_name]

print("\nAvailable V tracks:")
for name in v_track_names:
    print(" ", name)


# ------------------------------------------------------------
# LOAD AUDIO
# ------------------------------------------------------------

print("\nLoading audio files...")

# Load N track
n_path = os.path.join(folder_path, n_track_name)
n_audio, sr = librosa.load(n_path, sr=sample_rate)

# Load all V tracks
v_audio_map = {}
for file_name in v_track_names:
    file_path = os.path.join(folder_path, file_name)
    audio, _ = librosa.load(file_path, sr=sample_rate)
    v_audio_map[file_name] = audio

print(f"N track loaded: {n_track_name}")
print(f"Loaded {len(v_audio_map)} V tracks")


# ------------------------------------------------------------
# CREATE OUTPUT BUFFER
# IMPORTANT: start with the full N track
# ------------------------------------------------------------

output_audio = n_audio.astype(np.float32).copy()


# ------------------------------------------------------------
# RENDER REPLACEMENTS
# ------------------------------------------------------------

print("\nReplacing matched regions...")

for i, match in enumerate(matches, start=1):
    print(f"\nProcessing match {i}...")

    n_start_sec = match["n_start_sec"]
    n_end_sec = match["n_end_sec"]
    v_track = match["v_track"]
    v_start_sec = match["v_start_sec"]
    v_end_sec = match["v_end_sec"]
    score = match["score"]

    # Safety check: chosen N track should not also be used as a V source
    if v_track == n_track_name:
        print(f"  Skipped: V track '{v_track}' is the currently selected N track.")
        continue

    if v_track not in v_audio_map:
        print(f"  Skipped: V track '{v_track}' not found in V track list.")
        continue

    # Work out N region placement
    n_start_sample = time_to_samples(n_start_sec, sr)
    n_end_sample = time_to_samples(n_end_sec, sr)
    target_length = n_end_sample - n_start_sample

    if target_length <= 0:
        print("  Skipped: target length is zero or negative.")
        continue

    if n_start_sample >= len(output_audio):
        print("  Skipped: N start is beyond the end of the track.")
        continue

    # Cut V grain
    v_audio = v_audio_map[v_track]
    grain = extract_segment(v_audio, sr, v_start_sec, v_end_sec)

    if len(grain) == 0:
        print("  Skipped: extracted V grain is empty.")
        continue

    # Resize grain to match the N region length
    grain = resample_grain_to_length(grain, target_length)

    # Prevent writing beyond the end of the output buffer
    write_end = min(n_start_sample + len(grain), len(output_audio))
    actual_length = write_end - n_start_sample

    if actual_length <= 0:
        print("  Skipped: no writable output region.")
        continue

    grain = grain[:actual_length]

    # Get original N region
    original_region = output_audio[n_start_sample:write_end].copy()

    # Crossfade replacement grain with original N region at boundaries
    replaced_region = apply_edge_crossfade(
        original_region,
        grain,
        fade_length=fade_length_samples
    )

    # Write replaced region back into output
    output_audio[n_start_sample:write_end] = replaced_region

    print(f"  Replaced N region: {n_start_sec:.2f}s to {n_end_sec:.2f}s")
    print(f"  With V source:      {v_track}")
    print(f"  V region:           {v_start_sec:.2f}s to {v_end_sec:.2f}s")
    print(f"  Score:              {score:.3f}")

# ------------------------------------------------------------
# NORMALIZE AND SAVE
# ------------------------------------------------------------

print("\nNormalizing output...")
output_audio = normalize_audio(output_audio)

output_path = os.path.join(output_folder, output_file_name)
sf.write(output_path, output_audio, sr)

print("\nDone.")
print(f"Output written to: {output_path}")