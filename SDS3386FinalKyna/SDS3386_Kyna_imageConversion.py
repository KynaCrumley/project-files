import SimpleITK as sitk
import os
import re

'''Resample'''
def resample_image(image, new_spacing):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


'''remove artifacts for each image proccessed.
This is assuming the artifact has a significantly different intensity than the brain,
which is not be reallyyyy the case'''

def remove_artifacts(image):
    
    # we can calculate the threshold using Otsu's method
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.Execute(image)
    threshold = otsu_filter.GetThreshold()
    
    # Create an image filled with the threshold value + make sure to match the size and pixel type
    threshold_image = sitk.Image(image.GetSize(), image.GetPixelIDValue())
    threshold_image.CopyInformation(image)
    threshold_image += threshold  # Broadcasting the threshold value to all pixels
    
    #make a binary 'mask' where true values indicate the presence of the brain
    brain_mask = sitk.Greater(image, threshold_image)
    
    #Apply the mask to the image
    image = sitk.Mask(image, brain_mask, outsideValue=0)

    return image




def convert_dicom_to_nifti(dicom_folder, output_folder, target_spacing=1.2):

    
    #Make a list of all DICOM file paths in the folder
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)
    
    # Exclude the first 15 and the last 15 images (some files start and end with 15 slices of random noise, might be what's creating artifacts)
    #For simplicity and consistency, do the exclusion for all folders
    dicom_names = dicom_names[15:-15]

    #Check if there are DICOM files in the folder
    if not dicom_names:
        print(f"No DICOM files found in the folder: {dicom_folder}")
        return

    #Initialize the reader
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)

    #Execute the reader (to load the DICOM series)
    image = reader.Execute()

    # Resample if we need (i.e. if the spacing doesn't match the target spacing)
    current_spacing = image.GetSpacing()
    if current_spacing[2] != target_spacing: #should i explicitly set target spacing to 1.2? 
        new_spacing = (current_spacing[0], current_spacing[1], target_spacing)
        image = resample_image(image, new_spacing)

    #PermuteAxes to change the axes of the data
    image = sitk.PermuteAxes(image, [2, 1, 0])

    #remove artifacts
    #image = remove_artifacts(image)

    # Extracting the name for the NIFTI file from one of the DICOM files in the folder (naming is consistent throughout a folder)
    sample_name = os.path.basename(dicom_names[0])
    parts = sample_name.split('_')
    nifti_name = '_'.join(parts[1:4] + [parts[9]] + parts[-1].split('.'))
    nifti_name = re.sub(r'\.dcm$', '', nifti_name) + '.nii.gz'

    # Specify the output file path for the nifti
    output_file_path = os.path.join(output_folder, nifti_name)

    # Write the image to the NIFTI file
    sitk.WriteImage(image, output_file_path)

    print(f"Conversion complete. NIFTI file saved as {output_file_path}")



#walk through directories, if finds dicom, then take all files in that folder and convert to one nifti
def traverse_directories(root_folder, output_folder):
    for subdir, dirs, files in os.walk(root_folder):
        if any(f.endswith('.dcm') for f in files):
            convert_dicom_to_nifti(subdir, output_folder)

root_dir = '/Users/kynacrumley/Desktop/Fall2023.nosync/ADNI_DATA/ADNI'
output_dir = '//Users/kynacrumley/Desktop/Fall2023.nosync/ADNI_DATA/ADNI_NIITEST'
traverse_directories(root_dir, output_dir)
