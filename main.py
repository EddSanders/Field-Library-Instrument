import os
import librosa
import sounddevice as sd
import threading  # allows us to listen for input at the same time as playing audio

# path to your audio folder
folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

# get all files in the folder
files = os.listdir(folder)

# create a list to store only mp3 files
audio_files = []
for f in files:
    if f.endswith(".mp3"):  # only keep mp3 files
        audio_files.append(f)

audio_files.sort()  # sort files so they appear in order

# show files with numbers so user can choose
for i, f in enumerate(audio_files):
    print(i, f)

# user chooses a file by number
choice = int(input("Choose a number: "))

# check if choice is valid
if choice < 0 or choice >= len(audio_files):
    print("Invalid choice")
    exit()

# ask if user wants looping
loop_choice = input("Type loop to loop playback, or press Enter for normal playback: ")

# get the selected file and build full path
chosen_file = audio_files[choice]
full_path = os.path.join(folder, chosen_file)

# load audio file into memory
audio, sr = librosa.load(full_path)

print("Loaded successfully")
print("Playing:", chosen_file)
print("Duration:", len(audio) / sr)
print("Max amplitude:", max(audio))

# variable to track if stop has been requested
stop_requested = False

# function that listens for "stop" command
def stop_listener():
    global stop_requested  # allows us to modify the variable outside the function
    while True:
        command = input("Type 'stop' and press Enter to stop playback: ")
        if command.lower() == "stop":
            stop_requested = True  # tell main program to stop looping
            sd.stop()  # immediately stop audio
            break

# start the listener in a separate thread (runs alongside main code)
listener_thread = threading.Thread(target=stop_listener, daemon=True)
listener_thread.start()

# playback logic
if loop_choice.lower() == "loop":
    while not stop_requested:  # keep looping until stop is requested
        sd.play(audio, sr)
        sd.wait()  # wait until playback finishes
else:
    sd.play(audio, sr)
    sd.wait()

print("Playback finished")