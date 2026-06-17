# Please modify the species in config as correspond
#
python infer_generate_spectra.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_test.csv \
  --stats-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/avocado_groupA_globalD// \
  --experiment-dir G4_segmented_prospect_residual/ \
  --experiment-prefix vineyard \
  --stages auto \
  --output-dir ~/Results/pix2spectral_inference/avocadoModel/vineyard/test

python infer_generate_spectra.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --stats-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/avocado_groupA_globalD/ \
  --experiment-dir G4_segmented_prospect_residual/ \
  --experiment-prefix vineyard \
  --stages auto \
  --output-dir ~/Results/pix2spectral_inference/avocadomodel/vineyard/train

python infer_generate_spectra.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_val.csv \
  --stats-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/avocado_groupA_globalD/ \
  --experiment-dir G4_segmented_prospect_residual/ \
  --experiment-prefix vineyard \
  --stages auto \
  --output-dir ~/Results/pix2spectral_inference/avocadoModel/vineyard/val

# ---
# Assign FMC to the generated spectra
# ---

# Avocado
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/avocado/train/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
#  --output-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/avocado/val/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv \
#  --output-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/avocado/test/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
#  --output-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
#  --strict
## Olive
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/olive/train/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train.csv \
#  --output-csv ~/Results/pix2spectral_inference/olive/train/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/olive/val/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_val.csv \
#  --output-csv ~/Results/pix2spectral_inference/olive/val/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/olive/test/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_test.csv \
#  --output-csv ~/Results/pix2spectral_inference/olive/test/propectParams_with_FMC_d.csv \
#  --strict
#
## Vineyard
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/vineyard/train/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
#  --output-csv ~/Results/pix2spectral_inference/vineyard/train/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/vineyard/val/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_val.csv \
#  --output-csv ~/Results/pix2spectral_inference/vineyard/val/propectParams_with_FMC_d.csv \
#  --strict
#python assign_fmc_to_generated_spectra.py \
#  --generated-csv ~/Results/pix2spectral_inference/vineyard/test/prospect_parameters.csv --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_test.csv \
#  --output-csv ~/Results/pix2spectral_inference/vineyard/test/propectParams_with_FMC_d.csv \
#  --strict
