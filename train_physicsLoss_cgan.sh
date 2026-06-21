# python balance_trainingSet_by_fmc.py \
#   --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
#   --output-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train_balanced_FMC.csv \
#   --target-column FMC_d \
#   --n-bins 10 \
#   --target-per-bin min \
#   --random-state 42 \
#   --debug \
#   --debug-spectra \
#   --debug-images \
#   --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/"
# python balance_trainingSet_by_fmc.py \
#   --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train.csv \
#   --output-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train_balanced_FMC.csv \
#   --target-column FMC_d \
#   --n-bins 10 \
#   --target-per-bin min \
#   --random-state 42 \
#   --debug \
#   --debug-spectra \
#   --debug-images \
#   --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/"
#
# python balance_trainingSet_by_fmc.py \
#   --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train.csv \
#   --output-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train_balanced_FMC.csv \
#   --target-column FMC_d \
#   --n-bins 10 \
#   --target-per-bin min \
#   --random-state 42 \
#   --debug \
#   --debug-spectra \
#   --debug-images \
#   --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/"
#

#python run_cgan_3fold_groupkfold.py \
#  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train_val.csv \
#  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
#  --train-script train_with_physics_losses_cgan.py \
#  --output-root ~/Results/pix2spectral_cgan/avocado/3fold_64x64 \
#  --experiment-prefix avocado_cgan --species-filter avocado \
#  --group-column auto \
#  --n-splits 3 \
#  --batch-size 2 \
#  --num-epochs 200 \
#  --num-workers 0 \
#  --max-patches-per-band 90 \
#  --stop-on-failure

python run_cgan_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
  --train-script train_with_physics_losses_cgan.py \
  --output-root ~/Results/pix2spectral_cgan/olive/3fold_64x64 \
  --experiment-prefix olive_cgan --species-filter olive \
  --group-column auto \
  --n-splits 3 \
  --batch-size 2 \
  --num-epochs 100 \
  --num-workers 0 \
  --max-patches-per-band 90 \
  --stop-on-failure

python run_cgan_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
  --train-script train_with_physics_losses_cgan.py \
  --output-root ~/Results/pix2spectral_cgan/vineyard/3fold_64x64 \
  --experiment-prefix vineyard_cgan --species-filter vineyard \
  --group-column auto \
  --n-splits 3 \
  --batch-size 2 \
  --num-epochs 100 \
  --num-workers 0 \
  --max-patches-per-band 90 \
  --stop-on-failure

#python run_all_stage_experiments.py \
#  --train-script train_with_physics_losses_cgan.py \
#  --results-dir ~/Results/pix2spectral_final/Avocado/ \
#  --experiment-prefix avocado \
#  --stages fresh stage1 stage2 stage3 dry

#python run_all_stage_experiments.py \
#  --train-script train_with_physics_losses_cgan.py \
#  --results-dir ~/Results/pix2spectral_cgan \
#  --experiment-prefix avocado_cgan \
#  --stages fresh stage1 stage2 stage3 dry all \
#  --encoder-mode separate \
#  --normalization-scope global_band \
#  --num-workers 0 \
#  --batch-size 2 \
#  --stop-on-failure
