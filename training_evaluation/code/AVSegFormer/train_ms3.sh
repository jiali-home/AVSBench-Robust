CUDA_VISIBLE_DEVICES=0 python scripts/ms3/train.py "config/ms3/ood_AVSegFormer_pvt2_ms3.py"\
 --log_dir 'path_to_log_dir' --checkpoint_dir 'path_to_checkpoint_dir' --misaligned_percentage 10 --model AVSegFormer_robust
