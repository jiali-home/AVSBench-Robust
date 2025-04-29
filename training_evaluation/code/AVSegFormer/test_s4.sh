

# # s4: model on original data
echo "Testing S4 model on original data..."
python scripts/s4/test.py "config/s4/AVSegFormer_pvt2_s4.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file

echo "Testing S4 model on misaligned data..."
python scripts/s4/test.py "config/s4/Misaligned_AVSegFormer_pvt2_s4.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file


echo "Testing S4 model on noise data..."
python scripts/s4/test.py "config/s4/Noise_AVSegFormer_pvt2_s4.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file

echo "Testing S4 model on slience data..."
python scripts/s4/test.py "config/s4/Slience_AVSegFormer_pvt2_s4.py" "path_to_model" \
    --save_dir 'path_to_save_dir' \
    --save_pred_mask --model AVSegFormer_classifier >path_to_log_file

