def generate_script(num_exp):
    import random
    script = \
    """#!/bin/tcsh
    set DATASET="pheme"
    set LABEL_KINDS=2
    set FEATURE_DIM=768

    set DATA_DIR="./data/$DATASET/"
    set MODEL_DIR="./model/$DATASET/"
    set EPOCHS=1000
    set TRAIN_RATIO=0.72
    set VAL_RATIO=0.18
    set TEST_RATIO=0.1
    """
    run = \
    """
    set MODEL_NAME="$DATASET-rd-h$N_LAYERS-k$NEAREST_NEIGHBOR_K-dr$DROPOUT-wd$WEIGHT_DECAY-lr$LEARNING_RATE-ep$EPOCHS"
    python3 HMGNN.py --label_kinds $LABEL_KINDS --epochs $EPOCHS --data_dir $DATA_DIR --model_dir $MODEL_DIR \
        --feature_dim $FEATURE_DIM --nearest_neighbor_K $NEAREST_NEIGHBOR_K --model_name $MODEL_NAME --model_version 3 \
        --learning_rate $LEARNING_RATE --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
        --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
        --n_layers $N_LAYERS
    """
    parameters = {
        '$N_LAYERS': [1, 2, 3, 4, 5],
        '$NEAREST_NEIGHBOR_K': [7**i for i in range(6)],
        '$DROPOUT': [0.1*i for i in range(10)],
        '$WEIGHT_DECAY': [0.3**i for i in range(8)],
        '$LEARNING_RATE': [0.0001, 0.0005, 0.001, 0.005, 0.1],
    }
    # in total 5*6*10*8*5=12000 combinations -> run 1000
    for i in range(num_exp):
        exp = run
        for k, v in parameters.items():
            exp = exp.replace(k, str(random.sample(v, 1)[0]))
        script += exp
    with open('search.sh', 'w') as f:
        f.write(script)

generate_script(1000)