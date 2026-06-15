python run_groupA_physics_structure_experiments_updated.py \
  --train-script train_with_physics_losses.py \
  --eval-script evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
  --val-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/avocado_groupA_globalD \
  --species-filter Avocado \
  --experiment-prefix avocado \
  --stages fresh stage1 stage2 stage3 dry \
  --run-test-after-training \
  --stop-on-failure

python run_groupA_physics_structure_experiments_updated.py \
  --train-script train_with_physics_losses.py \
  --eval-script evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_test.csv \
  --val-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_val.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/olive_groupA_globalD \
  --species-filter Olive \
  --experiment-prefix olive \
  --stages fresh stage1 stage2 stage3 dry \
  --run-test-after-training \
  --stop-on-failure

python run_groupA_physics_structure_experiments_updated.py \
  --train-script train_with_physics_losses.py \
  --eval-script evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_test.csv \
  --val-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_val.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/vineyard_groupA_globalD \
  --species-filter Vineyard \
  --experiment-prefix vineyard \
  --stages fresh stage1 stage2 stage3 dry \
  --run-test-after-training \
  --stop-on-failure

python run_groupA_testing_only.py \
  --eval-script evaluate_test_set_export_spectra_smallN_groupA_fixed.py \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/avocado_groupA_globalD \
  --experiment-prefix avocado \
  --stages fresh stage1 stage2 stage3 dry

python run_groupA_testing_only.py \
  --eval-script evaluate_test_set_export_spectra_smallN_groupA_fixed.py \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train.csv \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_test.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/olive_groupA_globalD \
  --experiment-prefix olive \
  --stages fresh stage1 stage2 stage3 dry

python run_groupA_testing_only.py \
  --eval-script evaluate_test_set_export_spectra_smallN_groupA_fixed.py \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_test.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral/vineyard_groupA_globalD \
  --experiment-prefix vineyard \
  --stages fresh stage1 stage2 stage3 dry
