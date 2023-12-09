import os
import numpy as np

# Specify the directory containing the .npy files
directory_path = '/Users/cu/Documents/7th sem/cvlab/emojigame/FaceData'

# Define the emotions you are interested in
target_emotions = ['happy', 'sad', 'angry', 'surprise']

# Get a list of all files in the directory
all_files = os.listdir(directory_path)

# Initialize a dictionary to store data for each emotion
emotion_data = {emotion: [] for emotion in target_emotions}

# Group files based on emotions
for file in all_files:
    for emotion in target_emotions:
        if emotion in file and file.endswith('.npy'):
            emotion_data[emotion].append(np.load(os.path.join(directory_path, file)))

# Check if there are any matching emotion files
if any(emotion_data.values()):
    # Concatenate data for each emotion
    for emotion, data_list in emotion_data.items():
        if data_list:
            concatenated_data = np.concatenate(data_list, axis=0)

            # Save the concatenated data with a filename reflecting the emotion
            output_filename = f'{emotion}.npy'
            np.save(os.path.join(directory_path, output_filename), concatenated_data)
            print(f"Concatenation and save successful for {emotion}. Saved as {output_filename}")
else:
    print("No matching emotion files found in the directory.")
