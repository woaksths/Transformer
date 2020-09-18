# Transformer

## Install 
    $python3 -m venv venv
    $source venv/bin/activate
    $pip install -r requirements.txt

## Dataset
    $python scripts/generate_toy_data.py

## Usage
    #Before running this command, check the training option via $python train.py --help
    $python train.py --train_path $TRAIN_PATH --dev_path $DEV_PATH 
    $python test.py --checkpoint $TRAINED_MODEL --test_path $TEST_PATH
