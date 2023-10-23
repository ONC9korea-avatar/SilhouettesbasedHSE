# SilhouettesbasedHSE

## Install Environment
```
conda env create --file environment.yaml
```

## Dataset-generation
For more information, please refer to the [Dataset-generation README.md](dataset-generation/README.md)

## Prerequisites
1. Prepare train dataset, train dataset
2. Activate Conda
    ```
    conda activate HSE
    ```

## Train
1. Write the training configuration file,. You can refer to the `./config/hse/example.yaml`.
1. Run `./train.py`
    ```
    python -m train config/hse/example.yaml
    ```
    - The checkpoints is saved in `checkpoint_path` of the configuration file.
    
## Test
1. Run `./test.ipynb`

### TODO
- Save Dataset Result Configuration 