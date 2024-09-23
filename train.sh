CUDA_VISIBLE_DEVICES=1 python perspective_train.py --network mobilev3 --training_dataset licenseplate/train/label.txt \
    --save_folder mobilenet3_2/ --freeze_box_head True --lr 1e-2
    
