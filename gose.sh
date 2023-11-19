lang=$1
TRANSFORMERS_OFFLINE=1  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 21311  run_xfun_re.py \
        --model_name_or_path lilt-infoxlm-base \
        --tokenizer_name xlm-roberta-base \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1\
        --evaluation_strategy steps \
        --eval_steps 500 \
        --warmup_ratio 0.1 \
        --fp16 \
        \
        --use_trained_model False\
        --output_dir checkpoints/\
        --do_train \
        --do_eval \
        --lang ${lang}\
        --max_steps 40000 \
        --save_steps 40000 \
        --learning_rate  6.25e-6 \
        \
        --rounds 5\
        --decoder_name gose\
        --use_attn True\
        --use_prefix True\
        --use_gate True\
        --attn_lr 10\

echo $lang
echo  over