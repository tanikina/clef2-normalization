#!/bin/sh
export HOME=/netscratch/{username}/clef2-normalization
cd /netscratch/{username}/clef2-normalization
python -m pip install --upgrade pip
pip install wandb
pip install datasets
pip install evaluate
pip install nltk
python -c "import nltk; nltk.download('punkt')"
pip install pandas
pip install numpy
pip install transformers==4.47.1
pip install accelerate==1.2.1
export WANDB_API_KEY={WANDB_KEY}
wandb login
cd src/baselines
echo 1e-3-seed1
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-3 \
--seed=1
echo 1e-3-seed2
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-3 \
--seed=2
echo 1e-3-seed3
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-3 \
--seed=3
echo 5e-4-seed1
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=5e-4 \
--seed=1
echo 5e-4-seed2
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=5e-4 \
--seed=2
echo 5e-4-seed3
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=5e-4 \
--seed=3
echo 3e-4-seed1
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=3e-4 \
--seed=1
echo 3e-4-seed2
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=3e-4 \
--seed=2
echo 3e-4-seed3
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=3e-4 \
--seed=3
echo 1e-4-seed1
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-4 \
--seed=1
echo 1e-4-seed2
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-4 \
--seed=2
echo 1e-4-seed3
python official_baseline.py --language="English" \
--model_checkpoint="google/umt5-base" \
--train_data_path="../../data/train/train-eng.csv" \
--val_data_path="../../data/dev/dev-eng.csv" \
--max_epochs=20 \
--learning_rate=1e-4 \
--seed=3
