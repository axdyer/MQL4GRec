Dataset=Instruments

OUTPUT_DIR=../data/$Dataset

python -u generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path log/$Dataset/llama_256/best_collision_model.pth \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb.json \
  --data_path ../data/ \
  --embedding_file .emb-llama-td.npy


python -u generate_indices_distance.py \
    --dataset $Dataset \
    --device cuda:0 \
    --ckpt_path log/$Dataset/ViT-L-14_256/best_collision_model.pth \
    --output_dir $OUTPUT_DIR \
    --output_file ${Dataset}.index_vitemb.json \
    --content image \
    --data_path ../data/ \
    --embedding_file .emb-ViT-L-14.npy
