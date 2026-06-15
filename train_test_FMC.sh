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

python train_lwc_regressors_from_generated_spectra.py \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
  --val-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
  --target-column FMC_d \
  --species Avocado \
  --output-dir ~/Results/lwc_regression/avocado_real_spectra

python train_lwc_regressors_from_generated_spectra.py \
  --train-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
  --spectrum-column params_json \
  --target-column FMC_d \
  --species Avocado \
  --output-dir ~/Results/lwc_regression/avocado_grid_norm_400_1000 \
  --save-models

--grid_search \
  --grid-size medium

python train_lwc_regressors_from_generated_spectra_allleaf_norm_wlrange_fixed.py \
  --train-csv ~/Results/pix2spectral_inference/avocado/train/generated_spectra_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/avocado/val/generated_spectra_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/avocado/test/generated_spectra_with_FMC_d.csv \
  --target-column FMC_d \
  --species Avocado \
  --output-dir ~/Results/lwc_regression/avocado --wl-min 400 \
  --wl-max 2500 \
  --save-models

python train_lwc_regressors_from_prospect_params.py \
  --train-csv ~/Results/pix2spectral_inference/avocado/train/propectParams_with_FMC_d.csv \
  --val-csv ~/Results/pix2spectral_inference/avocado/val/propectParams_with_FMC_d.csv \
  --test-csv ~/Results/pix2spectral_inference/avocado/test/propectParams_with_FMC_d.csv \
  --target-column FMC_d \
  --species Avocado \
  --param-feature-mode flatten \
  --output-dir ~/Results/lwc_regression/avocado_prospect_params \
  --save-models
