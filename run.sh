#!/bin/tcsh

# foreach WEIGHT_DECAY (0.05 0.1)  # 0.05
    # foreach LEARNING_RATE (0.01 0.001)  # 0.001
        # foreach NEAREST_NEIGHBOR_K (3 7 11)

        set DATASET="pheme"
        # set DATASET="politifact"
        # set DATASET="buzzfeed"
        set LABEL_KINDS=2
        if ($DATASET == "politifact") then
            # set FEATURE_DIM=773
            # set BERT_IN="/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset/politifact"
            # set FEATURE_DIM=768
            set FEATURE_DIM=1438
        else if ($DATASET == "pheme") then
            # set FEATURE_DIM=775
            # set BERT_IN="/rwproject/kdd-db/20-rayw1/pheme-figshare"
            # set FEATURE_DIM=768
            set FEATURE_DIM=1440
        else if ($DATASET == "buzzfeed") then
            # set FEATURE_DIM=799
            # set BERT_IN="/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle"
            set FEATURE_DIM=768
        endif

        set DATA_DIR="./data/$DATASET/vocab_other/"
        # set FEATURE_DIM=1433 ###############

        set MODEL_DIR="./model/$DATASET/"
        set TRAIN_RATIO=0.72
        set VAL_RATIO=0.18
        set TEST_RATIO=0.1

        set N_LAYERS=5
        set NEAREST_NEIGHBOR_K=10
        set DROPOUT=0.2
        # Weight for L2 loss on embedding matrix. Defaults to 5e-4
        set WEIGHT_DECAY=0.001
        set LEARNING_RATE=0.03
        set EPOCHS=200

        # set MODEL_NAME="$DATASET-bertin-k$NEAREST_NEIGHBOR_K-dr$DROPOUT-wd$WEIGHT_DECAY-lr$LEARNING_RATE-ep$EPOCHS"
        set MODEL_NAME="$DATASET-vocab_other-h$N_LAYERS-k$NEAREST_NEIGHBOR_K-dr$DROPOUT-wd$WEIGHT_DECAY-lr$LEARNING_RATE-ep$EPOCHS"

        python3 HMGNN.py --label_kinds $LABEL_KINDS --epochs $EPOCHS --data_dir $DATA_DIR --model_dir $MODEL_DIR \
            --feature_dim $FEATURE_DIM --nearest_neighbor_K $NEAREST_NEIGHBOR_K --model_name $MODEL_NAME --model_version 3 \
            --learning_rate $LEARNING_RATE --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
            --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
            --hidden1 64 --hidden2 32 --n_layers $N_LAYERS \
            --minimum_subgraph_size 5
            # --bert_in $BERT_IN
                # end
    # end
# end
