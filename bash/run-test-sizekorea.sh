dataset_path=/workspace/dataset-generation/sizekorea_dataset_generation/sample_points
output_path=/workspace/SilhouettesbasedHSE/output
gt_path=/workspace/dataset-generation/sizekorea_dataset_generation/obj
export PYTHONPATH=$PYTHONPATH:"/workspace/SilhouettesbasedHSE"

conda run -n HSE python demo_code/sizekorea_infer.py sizekorea $dataset_path $output_path
conda run -n avatar python demo_code/sizekorea_test.py sizekorea $output_path $gt_path