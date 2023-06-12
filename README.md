# SilhouettesbasedHSE

## Install Environment
```
conda env create --file environment.yaml
```

## Prerequisites
1. Prepare train dataset, train dataset
    ```
    dataset
    ├── train
    │   ├── sample_0
    │   │   ├── frontal.npy
    │   │   ├── lateral.npy
    │   │   └── shape.npy
    │   ├── sample_1
    │   ├── sample_2
    │   ├── sample_3
    |   └── ...
    └── test
        ├── sample_0
        │   ├── frontal.npy
        │   ├── lateral.npy
        │   └── shape.npy
        ├── sample_1
        ├── sample_2
        ├── sample_3
        └── ...
    ```
2. Activate Conda
    ```
    conda activate HSE
    ```

## Train
1. Run `./train/our_train.py` or `./train/our_train.ipynb`
    - The `repeat_data` function makes the starting and ending points of the data connected to each other.
    - After training, you can see checkpoints in `./train/model`

## Test
1. Run cells under `Inference on Test Set` in `./test.ipynb`
    - Set `model_name` to the path to the checkpoint file to test
    - Set `obj_path` to the path of `./{samplename}.obj` files of dataset
    - After inference, you can see results in `./test_results/` with `mesh_gt.obj` and `mesh_out.obj`
2. Run cells under `Calculate V2V` in `./test.ipynb`
    - Set `result_path` to the path of inferenced results
