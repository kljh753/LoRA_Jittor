#!/bin/bash

# 快速训练版本 - 限制数据量以适配RTX 3060快速训练
echo "=== 创建快速训练数据集 (RTX 3060优化版) ==="

# 设置数据量限制 (适合几分钟内完成训练)
TRAIN_LIMIT=1500    # 训练集限制为5000条
VALID_LIMIT=300    # 验证集限制为1000条  
TEST_LIMIT=300     # 测试集限制为1000条

echo "数据量设置: 训练集=${TRAIN_LIMIT}, 验证集=${VALID_LIMIT}, 测试集=${TEST_LIMIT}"

echo "creating e2e datasets..."
path=data/e2e
echo "train..."
python src/format_converting_e2e.py $path/train.txt $path/train_formatted.jsonl
head -n $TRAIN_LIMIT $path/train_formatted.jsonl > $path/train_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted_limited.jsonl --output $path/train.jsonl --add_bos --add_eos

echo "test..."
python src/format_converting_e2e.py $path/test.txt $path/test_formatted.jsonl
head -n $TEST_LIMIT $path/test_formatted.jsonl > $path/test_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted_limited.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_e2e.py $path/valid.txt $path/valid_formatted.jsonl
head -n $VALID_LIMIT $path/valid_formatted.jsonl > $path/valid_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted_limited.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "creating webnlg datasets..."
path=data/webnlg_challenge_2017
echo "train..."
python src/format_converting_webnlg.py $path/train.json $path/train_formatted.jsonl
head -n $TRAIN_LIMIT $path/train_formatted.jsonl > $path/train_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted_limited.jsonl --output $path/train.jsonl --add_bos --add_eos

echo "test..."
python src/format_converting_webnlg.py $path/test.json $path/test_formatted.jsonl
head -n $TEST_LIMIT $path/test_formatted.jsonl > $path/test_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted_limited.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_webnlg.py $path/dev.json $path/valid_formatted.jsonl
head -n $VALID_LIMIT $path/valid_formatted.jsonl > $path/valid_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted_limited.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "creating dart datasets..."
path=data/dart
echo "train..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-train.json data/dart/train_formatted.jsonl
head -n $TRAIN_LIMIT $path/train_formatted.jsonl > $path/train_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted_limited.jsonl --output $path/train.jsonl --add_bos --add_eos

echo "test..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-test.json data/dart/test_formatted.jsonl
head -n $TEST_LIMIT $path/test_formatted.jsonl > $path/test_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted_limited.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-dev.json data/dart/valid_formatted.jsonl
head -n $VALID_LIMIT $path/valid_formatted.jsonl > $path/valid_formatted_limited.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted_limited.jsonl --output $path/valid.jsonl --add_bos --add_eos

# 清理临时文件
echo "清理临时文件..."
find data/ -name "*_formatted_limited.jsonl" -delete