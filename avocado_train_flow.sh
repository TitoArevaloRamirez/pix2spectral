python test_flow2spectral.py \
    --test_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
    --checkpoint_path ~/Checkpoints/pix2spectral/Avocado/avocado_best.pt \
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage all \
    --test_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_test_prospect_data.npz \
    --n_samples_per_stage 5 \
    --sampling_steps 100 \
    --force_recompute_cache


python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage All \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log.txt \
    --epochs 6000 \
    --val_every 100 \
    --early_stop_patience 1200 \
    --hidden 64 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 24 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 10 \
    --force_recompute_cache

python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage no_fresh \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data_no_fresh.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data_no_fresh.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint_no_fresh.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best_no_fresh.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log_no_fresh.txt \
    --force_recompute_cache \
    --epochs 5000 \
    --val_every 100 \
    --early_stop_patience 1000 \
    --hidden 32 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 16 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 3

python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage no_stage1 \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data_no_stage1.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data_no_stage1.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint_no_stage1.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best_no_stage1.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log_no_stage1.txt \
    --force_recompute_cache \
    --epochs 5000 \
    --val_every 100 \
    --early_stop_patience 1000 \
    --hidden 32 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 16 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 3

python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage no_stage2 \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data_no_stage2.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data_no_stage2.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint_no_stage2.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best_no_stage2.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log_no_stage2.txt \
    --force_recompute_cache \
    --epochs 5000 \
    --val_every 100 \
    --early_stop_patience 1000 \
    --hidden 32 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 16 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 3

python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage no_stage3 \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data_no_stage3.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data_no_stage3.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint_no_stage3.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best_no_stage3.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log_no_stage3.txt \
    --force_recompute_cache \
    --epochs 5000 \
    --val_every 100 \
    --early_stop_patience 1000 \
    --hidden 32 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 16 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 3

python flow2spectral_conditioned.py \
    --train_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val_csv_path ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv\
    --root_dir "/media/usr3/TAR-Lab/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --species Avocado \
    --stage no_dry \
    --train_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_prospect_data_no_dry.npz \
    --val_cache_path ~/Checkpoints/pix2spectral/Avocado/cache/avocado_val_prospect_data_no_dry.npz \
    --save_path ~/Checkpoints/pix2spectral/Avocado/avocado_checkpoint_no_dry.pt \
    --best_save_path ~/Checkpoints/pix2spectral/Avocado/avocado_best_no_dry.pt \
    --log_path ~/Checkpoints/pix2spectral/Avocado/avocado_conditional_flow_log_no_dry.txt \
    --force_recompute_cache \
    --epochs 5000 \
    --val_every 100 \
    --early_stop_patience 1000 \
    --hidden 32 \
    --depth 2 \
    --time_dim 8 \
    --condition_dim 16 \
    --dropout 0.15 \
    --weight_decay 1e-3 \
    --min_delta 1e-4 \
    --val_repeats 3



#
# ----------------------------------------------------------
#

