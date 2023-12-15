import torch
import nibabel as nib
import pandas as pd
import os
import re
import sys
import skimage.transform as skTrans

# Initialize pretrained model
device = torch.device('cpu')

checkpoint_path = '/Users/kynacrumley/Desktop/Fall2023.nosync/SDS3386/Term_Project/age_expansion_8_model_low_loss.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device)
model_state_dict = checkpoint['state_dict']

# So that I can run the from models import NetWork as a .py in my terminal
module_path = '/Users/kynacrumley/Desktop/Fall2023.nosync/SDS3386/Term_Project'
sys.path.append(module_path)
from modelsv2 import NetWork

model = NetWork(in_channel=3, feat_dim=1024)  # should be 1024

# Load the reference sheet
ref_df = pd.read_csv('/Users/kynacrumley/Desktop/PTReferenceSheet.csv')

# Extract Features from all files in a folder.
results = []
root_path = '/Users/kynacrumley/Desktop/Fall2023.nosync/ADNI_DATA/ADNI1-3Raw/ADNI_Raw'

# Let's keep track of files that aren't processed correctly
unprocessed_files = []

# Regular expression to extract subject ID
id_pattern = re.compile(r'([A-Z]\d+)_ADNI_')

# Traverse directories and process .nii files
for subdir, dirs, files in os.walk(root_path):
    for file_name in files:
        if file_name.endswith('.nii'):
            print(f"Processing file: {file_name}")  # Print the file being processed

            match = id_pattern.match(file_name)
            if match:
                try:
                    unique_id = match.group(1)  # Unique ID (ex. I100021)

                    ref_info = ref_df[ref_df['UniqueID'] == unique_id]
                    if not ref_info.empty:
                        session_date = ref_info['SessionId'].iloc[0]
                        age = ref_info['EstimatedAge'].iloc[0]
                        age_tensor = torch.tensor([age], dtype=torch.float32).to(device)
                    else:
                        # If age is not found, set age_tensor to None
                        age_tensor = None
                        session_date = "Unknown"  # Handle unknown session date

                    image_path = os.path.join(subdir, file_name)

                    # Load image
                    image = nib.load(image_path)
                    data = image.get_fdata()
                    data = skTrans.resize(data, (3, 100, 100, 100), order=1, preserve_range=True)
                    data = (data - data.mean()) / data.std()
                    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

                    # Forward pass through the model
                    output = model(data, age_id=age_tensor).squeeze().detach().cpu().numpy()

                    # Create a dictionary for each image
                    image_features = {'UniqueID': unique_id, 'SessionId': session_date}
                    for i, value in enumerate(output):
                        feature_key = f'Feature_{i + 1}'
                        image_features[feature_key] = value

                    results.append(image_features)

                    print('Processed successfully:', file_name)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    unprocessed_files.append({'FileName': file_name, 'Error': str(e)})
            else:
                print(f"File name pattern does not match: {file_name}")
                unprocessed_files.append({'FileName': file_name, 'Error': 'Pattern Mismatch'})

# Create DataFrames from the results and unprocessed files
df = pd.DataFrame(results)
df_unprocessed = pd.DataFrame(unprocessed_files)

# Save to CSV
df.to_csv("ADNI123_051223KC_Features_Wide.csv", index=False)
df_unprocessed.to_csv("Unprocessed_Files.csv", index=False)
