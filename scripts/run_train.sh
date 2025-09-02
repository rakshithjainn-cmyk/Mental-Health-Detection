#!/usr/bin/env bash
python src/train.py --data_path data/sample.csv --model_name bert-base-uncased --output_dir models/bert-mh --epochs 3 --batch_size 16 --max_length 128
