accelerate launch --config_file ./config.yaml --main_process_port 32340 --gpu_ids 2, src/main_kdd.py --task pretrain --version full_run --ep 30 --num-pos 1 --num-negs 0 --label-pool-size 150  --bs 128 --cl-start 2 --cl-update 4 --loss-lambda -1 --fill-batch-gap 0