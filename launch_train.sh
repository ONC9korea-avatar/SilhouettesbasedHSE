export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NGPUS=7

cores=`nproc`
num_worker=20
export OMP_NUM_THREADS=$(($cores-$num_worker)) # you can change this value according to your number of cpu cores


#python -m torch.distributed.launch --nproc_per_node=$NGPUS 7.multi_seg_input_train.py

torchrun --nproc_per_node=$NGPUS 7.multi_seg_input_train.py