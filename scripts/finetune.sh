
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Tasks='seqrec,seqimage,item2image,image2item,fusionseqrec'
Valid_task=seqrec

Datasets='Instruments'
Per_device_batch_size=64
Max_his_len=20
Num_beams=20

load_model_name=./log/$Datasets/ckpt_b1024_lr1e-3_seqrec,seqimage/pretrain

OUTPUT_DIR=./log/$Datasets
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

torchrun --nproc_per_node=2 --master_port=2312 finetune.py \
    --data_path ./data/ \
    --dataset $Datasets \
    --output_dir $OUTPUT_DIR \
    --load_model_name $load_model_name \
    --per_device_batch_size $Per_device_batch_size \
    --learning_rate 5e-4 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len $Max_his_len \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task > $log_file

results_file=$OUTPUT_DIR/results_${Valid_task}_${Num_beams}.json
save_file=$OUTPUT_DIR/save_${Valid_task}_${Num_beams}.json

torchrun --nproc_per_node=1 --master_port=2310 test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams $Num_beams \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items > $log_file

# Also run test for seqimage task
results_file_image=$OUTPUT_DIR/results_seqimage_${Num_beams}.json
save_file_image=$OUTPUT_DIR/save_seqimage_${Num_beams}.json

torchrun --nproc_per_node=1 --master_port=2311 test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams $Num_beams \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task seqimage \
    --results_file $results_file_image \
    --save_file $save_file_image \
    --filter_items >> $log_file

python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path ./data/\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams $Num_beams

