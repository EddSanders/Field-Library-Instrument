import os

folder = r"C:\Users\edwar\OneDrive\Desktop\NVInitialResearch\FieldLibrary\NorthernPoolFrog"

files = os.listdir(folder)

print("Files found:")
for f in files:
    print(f)