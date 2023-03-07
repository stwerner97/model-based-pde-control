#!/bin/bash

PROJECT="KS-Offline-Evaluation-Dissipation-Model-Capacity"
ENV="KuramotoSivashinskyEnv-v0"
DATA="KSattractor.pl"

# Execution Parameters
SPLITS=5
TOTALS=( 0.5 )
TARGET=30
VAL=0.2
SEED=0

# Surrogate Model Type
FACTORY="KSAutoRegConvolutionalLSTM"

# Model & Training Parameters 
LOSS="MSELoss"
SURROGATE="{}"
TRAINING='{"tbtt": 1000000, "tau": 10, "batch_size": 64, "patience": 50}'
CURRICULUM='{}'
TRAINER='{"max_epochs": 150, "gradient_clip_val": 0.5}'


export PYTHONPATH="/workspace"

for total in "${TOTALS[@]}"
do
    # Add parameter "--untransformed" manually !
    MODEL='{"width_coefficient": 1.5}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    MODEL='{"width_coefficient": 2.0}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    MODEL='{"width_coefficient": 2.5}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY  --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    MODEL='{"width_coefficient": 3.0}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    MODEL='{"width_coefficient": 3.5}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    MODEL='{"width_coefficient": 4.0}'
    python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $TARGET --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
done