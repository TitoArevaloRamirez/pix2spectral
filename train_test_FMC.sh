#python train_lwc_regressors_from_generated_spectra.py \
#  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
#  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
#  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
#  --target-column FMC_d \
#  --species Avocado \
#  --output-dir ~/Results/lwc_regression/avocado_generated_spectra

#python train_lwc_regressors_from_generated_spectra_grid.py \
#  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
#  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
#  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
#  --target-column FMC_d \
#  --species Avocado \
#  --output-dir ~/Results/lwc_regression/avocado_grid \
#  --grid_search \
#  --grid-size medium \
#  --save-models
#

python train_lwc_regressors_from_generated_spectra_subplots.py \
  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
  --target-column FMC_d \
  --species Avocado \
  --output-dir ~/Results/lwc_regression/avocado_generated_spectra \
  --save-models

python train_lwc_regressors_from_generated_spectra_subplots.py \
  --train-csv ~/Results/pix2spectral_inference/olive/train/generated_spectra_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/olive/val/generated_spectra_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/olive/test/generated_spectra_with_FMC_d.csv \
  --target-column FMC_d \
  --species Olive \
  --output-dir ~/Results/lwc_regression/olive_generated_spectra \
  --save-models

python train_lwc_regressors_from_generated_spectra_subplots.py \
  --train-csv ~/Results/pix2spectral_inference/vineyard/train/generated_spectra_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/vineyard/val/generated_spectra_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/vineyard/test/generated_spectra_with_FMC_d.csv \
  --target-column FMC_d \
  --species Vineyard \
  --output-dir ~/Results/lwc_regression/vineyard_generated_spectra \
  --save-models

#python train_lwc_regressors_from_prospect_params.py \
#  --train-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
#  --val-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
#  --test-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
#  --target-column FMC_d \
#  --species Avocado \
#  --param-feature-mode flatten \
#  --output-dir ~/Results/lwc_regression/avocado_prospect_params \
#  --save-models

#python train_lwc_regressors_from_generated_spectra_fusion_features.py \
#  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
#  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
#  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
#  --train-params-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
#  --val-params-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
#  --test-params-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
#  --target-column FMC_d \
#  --species Avocado \
#  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
#  --output-dir ~/Results/lwc_regression/avocado_fusion_features \
#  --save-models \
#  --channel-columns blue:blue_basename
#
#python train_lwc_regressors_from_generated_spectra_fusion_features_v2.py \
#  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
#  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
#  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
#  --train-params-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
#  --val-params-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
#  --test-params-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
#  --target-column FMC_d \
#  --species Avocado \
#  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
#  --output-dir ~/Results/lwc_regression/avocado_fusion_features_v2 \
#  --save-models
