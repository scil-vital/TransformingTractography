
#########
# Choose study
#########
database_folder=MY_PATH


#########
# Organize your files.
#########

#########
# Create hdf5 dataset
# ** If you have already used Learn2track, you can reuse the same hdf5!
#########
space='rasmm'
name='my_hdf5_database'
mask="mask/*__mask_wm.nii.gz"

dwi_ml_ready_folder=$database_folder/derivatives/dwi_ml_ready
hdf5_folder=$database_folder/derivatives
config_file=my_config_file
training_subjs=file1.txt
validation_subjs=file2.txt
testing_subjs=file3.txt
tt_create_hdf5_dataset.py --force --name $name --std_mask $mask \
        --logging info --space $space --enforce_files_presence True \
        $dwi_ml_ready_folder $hdf5_folder $config_file \
        $training_subjs $validation_subjs $testing_subjs

