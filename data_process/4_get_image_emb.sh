
export CUDA_VISIBLE_DEVICES=1

python clip_feature.py \
    --image_root ./datasets/amazon18 \
    --save_root ./datasets/MQL4GRec \
    --model_cache_dir ./cache_models/clip \
    --dataset Instruments


