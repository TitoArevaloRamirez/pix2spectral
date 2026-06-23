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

#python run_cgan_3fold_groupkfold.py \
#  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train_val.csv \
#  --img-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
#  --train-script train_with_physics_losses_cgan.py \
#  --output-root ~/Results/pix2spectral_cgan/olive/3fold_64x64 \
#  --experiment-prefix olive_cgan --species-filter olive \
#  --group-column auto \
#  --n-splits 3 \
#  --batch-size 2 \
#  --num-epochs 100 \
#  --num-workers 0 \
#  --max-patches-per-band 90 \
#  --stop-on-failure
#
#python run_cgan_3fold_groupkfold.py \
#  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train_val.csv \
#  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
#  --train-script train_with_physics_losses_cgan.py \
#  --output-root ~/Results/pix2spectral_cgan/vineyard/3fold_64x64 \
#  --experiment-prefix vineyard_cgan --species-filter vineyard \
#  --group-column auto \
#  --n-splits 3 \
#  --batch-size 2 \
#  --num-epochs 100 \
#  --num-workers 0 \
#  --max-patches-per-band 90 \
#  --stop-on-failure

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
#
#
#
python run_cgan_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/MultispectralImages_resized_220_norm/" \
  --train-script train_with_physics_losses_mobilenetv3_fullleaf_clean.py \
  --output-root ~/Results/pix2spectral_mobilenetv3/avocado_freezeAll \
  --experiment-prefix avocado_mobilenetv3 --group-column auto \
  --n-splits 3 \
  --batch-size 2 \
  --num-epochs 100 \
  --num-workers 0 \
  --stop-on-failure

export SPECIES_FILTER=avocado
python run_mobilenetv3_fullleaf_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Avocado/MultispectralImages_resized_220_norm/" \
  --train-script train_with_physics_losses_mobilenetv3_fullleaf_clean.py \
  --output-root ~/Results/pix2spectral_mobilenetv3/avocado_freezeAll \
  --experiment-prefix avocado_mobilenetv3 \
  --species avocado \
  --group-column auto \
  --n-splits 3 \
  --full-image-size 220 \
  --batch-size 2 \
  --num-epochs 100 \
  --learning-rate 2e-05 \
  --mobilenet-token-dim 64 \
  --mobilenet-attention-heads 2 \
  --mobilenet-dropout 0.40 \
  --stage-aux-weight 0.02 \
  --lambda-mismatch 0.1 \
  --best-model-metric val_l1 \
  --early-stop-patience 20 \
  --early-stop-min-epochs 50 \
  --stream-output \
  --stop-on-failure

export SPECIES_FILTER=olive
python run_cgan_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/olive_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Olive/MultispectralImages_resized_220_norm/" \
  --train-script train_with_physics_losses_mobilenetv3_fullleaf_clean.py \
  --output-root ~/Results/pix2spectral_mobilenetv3/olive_freezeAll \
  --experiment-prefix olive_mobilenetv3 --group-column auto \
  --n-splits 3 \
  --batch-size 2 \
  --num-epochs 100 \
  --num-workers 0 \
  --stop-on-failure

python run_cgan_3fold_groupkfold.py \
  --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train_val.csv \
  --img-dir "/home/usr3/Data/EstradaDataset/Vineyard/MultispectralImages_resized_220_norm/" \
  --train-script train_with_physics_losses_mobilenetv3_fullleaf_clean.py \
  --output-root ~/Results/pix2spectral_mobilenetv3/vineyard_freezeAll \
  --experiment-prefix avocado_mobilenetv3 --group-column auto \
  --n-splits 3 \
  --batch-size 2 \
  --num-epochs 100 \
  --num-workers 0 \
  --stop-on-failure
