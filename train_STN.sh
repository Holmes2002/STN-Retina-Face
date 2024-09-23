CUDA_VISIBLE_DEVICES=1 python train_STN.py --root /home1/data/congvu/RetinaFace/STN_dataset_2 \
    --batch_size 64 --lr 5e-7 \
    --float16 True
# CUDA_VISIBLE_DEVICES=0 python inference.py --root /home/data/congvu/Face_Align/test_img --batch_size 4 --lr 2e-6