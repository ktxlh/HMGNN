#!/bin/tcsh
foreach DATASET ("pheme/" "politifact/")
    foreach NEAREST_NEIGHBOR_K (1 2 3 4 5 6 7 8 9 10)
        set LABEL_KINDS=2
        set EPOCHS=300
        set DATA_DIR="./data/$DATASET"
        set MODEL_DIR="./model/$DATASET"
        if ($DATASET == "pheme/") then
            set FEATURE_DIM=768
        else
            set FEATURE_DIM=1536
        endif
        set MODEL_NAME="GCN-k$NEAREST_NEIGHBOR_K"

        python3 HMGNN.py --label_kinds $LABEL_KINDS --epochs $EPOCHS --data_dir $DATA_DIR --model_dir $MODEL_DIR \
            --feature_dim $FEATURE_DIM --nearest_neighbor_K $NEAREST_NEIGHBOR_K --model_name $MODEL_NAME
    end
end
