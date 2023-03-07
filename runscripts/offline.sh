#!/bin/bash

PROJECT="KS-Offline-Evaluation-Dissipation"
ENV="KuramotoSivashinskyEnv-v0"
DATA="KSattractor.pl"

# Execution Parameters
SPLITS=5
TOTALS=( 0.9 0.8 0.6 0.5 0.3 0.2 )
TARGETS=( 30 )
VAL=0.2
SEED=0

# Surrogate Model Type
FACTORY="KSAutoRegConvolutionalLSTM"

# Model & Training Parameters 
LOSS="MSELoss"
MODEL="{}"
SURROGATE="{}"
TRAINING='{"tbtt": 1000000, "tau": 10, "batch_size": 64, "patience": 50}'
CURRICULUM='{"scheduler": "LinearScheduler", "steptype": "epoch", "start": 0, "stop": 100, "vmin": 25, "vmax": 50}'
TRAINER='{"max_epochs": 250, "gradient_clip_val": 0.5}'


export PYTHONPATH="/workspace"

for total in "${TOTALS[@]}"
do
    for target in "${TARGETS[@]}"
    do
        # Add parameter "--untransformed" manually !
        python3.8 pdecontrol/surrogates/evaluation/evaluate.py --project $PROJECT --splits $SPLITS --total $total --data $DATA --val $VAL --target_length $target --env_id $ENV --loss $LOSS --seed $SEED --factory $FACTORY --factory_module $MODULE --model "$MODEL" --surrogate "$SURROGATE" --training "$TRAINING" --curriculum "$CURRICULUM" --trainer "$TRAINER"
    done
done
