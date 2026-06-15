python evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
  --results-root ~/Results/pix2spectral \
  --experiment-prefix avocado \
  --experiment-dirs avocado_global avocado_segmented avocado_global_plus_segmented \
  --mode-labels global segmented global_plus_segmented \
  --stages fresh stage1 stage2 stage3 dry

python evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_test.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --results-root ~/Results/pix2spectral \
  --experiment-prefix vineyard \
  --experiment-dirs vineyard_global vineyard_segmented vineyard_global_plus_segmented \
  --mode-labels global segmented global_plus_segmented \
  --stages fresh stage1 stage2 stage3 dry

python evaluate_test_set_export_spectra_smallN.py \
  --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_test.csv \
  --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
  --results-root ~/Results/pix2spectral \
  --experiment-prefix olive \
  --experiment-dirs olive_global olive_segmented olive_global_plus_segmented \
  --mode-labels global segmented global_plus_segmented \
  --stages fresh stage1 stage2 stage3 dry
