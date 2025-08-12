#!/bin/bash

# WebNLG dataset Jittor version experiment script - Optimized Configuration
# Based on https://github.com/coder-yd/Lora_jittor best practices
# Process: train -> generate -> decode -> evaluate

echo "=== WebNLG Dataset Jittor Version LoRA Experiment (Optimized) ==="
echo "Start time: $(date)"

LOG_DIR="./logs/webnlg"

# Create log directories
mkdir -p $LOG_DIR

echo "=== Step 1: Train GPT-2 + LoRA ==="
python src/gpt2_ft.py \
    --train_data ./data/webnlg_challenge_2017/train.jsonl \
    --valid_data ./data/webnlg_challenge_2017/valid.jsonl \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 256 \
    --model_card gpt2.sm \
    --init_checkpoint ./pretrained_checkpoints/gpt2-pytorch_model.bin \
    --platform single \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 3 \
    --save_interval 1000 \
    --log_interval 20 \
    --eval_interval 200 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_SM/webnlg \
    --random_seed 100 \
    2>&1 | tee $LOG_DIR/train.log

echo "=== Step 2: Generate Text ==="
python src/gpt2_beam.py \
    --data ./data/webnlg_challenge_2017/test.jsonl \
    --batch_size 1 \
    --seq_len 256 \
    --eval_len 64 \
    --model_card gpt2.sm \
    --init_checkpoint ./trained_models/GPT2_M/webnlg/model.1000.pkl \
    --platform single \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 5 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 3 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_SM/webnlg \
    --output_file predict.1000.b5p08.jsonl \
    2>&1 | tee $LOG_DIR/inference.log

echo "=== Step 3: Decode Results ==="
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_SM/webnlg/predict.1000.b5p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 4 \
    --output_ref_file ./eval/data/references_webnlg_jittor \
    --output_pred_file ./eval/data/hypothesis_webnlg_jittor \
    --tokenize --lower \
    2>&1 | tee $LOG_DIR/decode.log

echo "=== Step 4: Evaluate Metrics ==="
if [ -d "./eval/GenerationEval/" ]; then
    cd ./eval/GenerationEval/
    python eval.py \
        -R data/references_webnlg_jittor/reference \
        -H data/hypothesis_webnlg_jittor \
        -nr 4 \
        -m bleu,meteor,ter \
        2>&1 | tee ../../$LOG_DIR/eval.log
    cd ../../
else
    echo "Evaluation directory does not exist, skipping evaluation step"
fi

echo "=== WebNLG Jittor Version Experiment Complete ==="
echo "End time: $(date)"
echo "Logs saved in: $LOG_DIR"