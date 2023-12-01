ROUNDS=$1
PORT=$2
GLOBAL=$3
save="checkpoints"/"multi"/"lilt"_"$ROUNDS"_"$GLOBAL"
# save="$lang"_"attn_mean"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port $PORT  run_xfun_re.py \
        --model_name_or_path lilt-infoxlm-base \
        --tokenizer_name xlm-roberta-base \
        --output_dir $save \
        --do_train \
        --additional_langs all \
        --max_steps 160000 \
        --save_steps 160000  \
        --per_device_train_batch_size 2 \
        --evaluation_strategy steps \
        --eval_steps 200000 \
        --learning_rate  6.25e-6 \
        --warmup_ratio 0.1 \
        --fp16 \
        \
        --decoder_name gose \
        --backbone_name lilt \
        --pooling_mode mean \
        --rounds $ROUNDS \
        --attn_lr 10 \
        --global_token_num $GLOBAL \
        \
        --use_gam \
        --use_gate \
        --use_prefix \
        \
        --overwrite_output_dir \
        --load_best_model_at_end=True \
        --save_total_limit 1 \
        --metric_for_best_model  f1 \