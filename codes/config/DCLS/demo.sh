

# for single GPU
# python train.py -opt=options/setting1/train_setting1_x4.yml

# for x2
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4312 train.py -opt=options/setting1/train/train_setting1_x2.yml --launcher pytorch

# for x3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4313 train.py -opt=options/setting1/train/train_setting1_x3.yml --launcher pytorch

# for x4
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4314 train.py -opt=options/setting1/train/train_setting1_x4.yml --launcher pytorch


# aniso x2
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4322 train.py -opt=options/setting2/train/train_setting2_x2.yml --launcher pytorch

# aniso x4
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4324 train.py -opt=options/setting2/train/train_setting2_x4.yml --launcher pytorch


### testing ###
# python test.py -opt=options/setting1/test/test_setting1_x2.yml

python test.py -opt=options/setting2/test/test_setting2_x2.yml