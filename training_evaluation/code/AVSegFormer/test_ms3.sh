
# ============================================
# ms3: model on original data
echo "Testing MS3 model on original data..."
python scripts/ms3/test.py "config/ms3/AVSegFormer_pvt2_ms3.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file


# ms3: model on misaligned data
echo "Testing MS3 model on misaligned data..."
python scripts/ms3/test.py "config/ms3/Misaligned_AVSegFormer_pvt2_ms3.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file

# ms3: model on noise data
echo "Testing MS3 model on noise data..."
python scripts/ms3/test.py "config/ms3/Noise_AVSegFormer_pvt2_ms3.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file


# ms3: model on slience data
echo "Testing MS3 model on slience data..."
python scripts/ms3/test.py "config/ms3/Slience_AVSegFormer_pvt2_ms3.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file


# ============================================


