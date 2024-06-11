
conda env create -n kdd_aqa --file=environment.yml

##Stage 1 training command for single GPU

accelerate launch --config_file ./config.yaml --main_process_port 32342 --gpu_ids 0 --num_processes 1 src/main.py --task pretrain --version Stage1 --ep 50 --num-negs 0 --label-pool-size 500 --bs 256 --add-dual-loss --lr 5e-5 --cl-start 5 --cl-update 5 

##Stage 2 training comman for single GPU

accelerate launch --config_file ./config.yaml --main_process_port 32342 --gpu_ids 0 --num_processes 1 src/main.py --task train --version Stage2 --ep 50 --num-negs 1 --label-pool-size 210 --bs 64 --lr 5e-5 --cl-start 5 --cl-update 5 --fill-batch-gap 0 --lm model_stage1.pth --load-from-pt
