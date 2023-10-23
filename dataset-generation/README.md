# Human Shape Estimation Dataset Generation

This Python script is designed to generate a dataset for Human Shape Estimation (HSE). It uses the SMPL model for reconstructing human shapes and produces silhouettes, sample points, and other data for the dataset.

## Requirements
- HSE environment
  - Python 3.7 or higher
  - OpenCV (cv2)
  - NumPy
  - PyTorch
  - tqdm
- SMPL model. refer to [SMPL_README](../SMPL/README.md)
- Config YAML file for defining dataset parameters

## Configuration

Before running the script, you need to specify the dataset generation parameters in a YAML configuration file (e.g., `config.yaml`). Below is an example of a configuration file:

```yaml
betas_path: /workspace/SilhouettesbasedHSE/dataset-generation/augmented_betas.npy
pose:
  per_beta: 3
  type: random_FL # random_L # T-pose # A-pose
seed: 42
npz:
  - beta
  - pose
  - vertices
  - sample_point

smpl_reconstruction:
  model_path: /workspace/SilhouettesbasedHSE/SMPL/model.pkl

camera:
  distance: 1.8
  pinhole:
    run: true

silhouette:
  image_width: 500
  image_height: 600
  max_human_height: 2.30
  save_png: true
  save_path: /workspace/datasets/dataset_hse

sample_points:
  run_sample_points: true
  num_sample_points: 648
```

- `betas_path`: Path to the beta values used for shape estimation.
- `pose`: Pose generation settings.
    - `per_beta`: The number of poses generated per beta value.
    - `type`: The type of pose generation (options: random_FL, random_L, T-pose, A-pose).
- `smpl_reconstruction`: Path to the SMPL model for reconstruction.
- `camera`: Camera settings for rendering silhouettes.
- `silhouette`: Settings for generating silhouettes.
    - `save_path`: The directory path where the generated silhouette images will be saved. Silhouette images will be stored in this directory as part of the dataset generation process. You should specify the desired directory location for saving silhouette images.
- `sample_points`: Settings for generating sample points.

## Running the Script
Please run the python command from the root directory of the repository.
To run the script, use the following command:
```
cd ..
python -m dataset-generation.generation ./config/dataset-generation/example.yaml
```
Replace config.yaml with the path to your configuration file.

## Output
The script will generate a dataset in the form of a compressed numpy file (`dataset.npz`). The file will contain various data, including betas, poses, vertices, sample points, etc. If you don't want to save a properties of `dataset.npz`, you should delete the enum of `npz` in your configuration file.