# python detect.py --image "/home/data/congvu/Retinaface_Ghost/home1/data/dongtrinh/workspace/Retinaface_Ghost/data/licenseplate/train/images/10148.jpg" \
#     -m ./mobilev3/mobilev3_epoch_320.pth --network mobilev3 --vis_thres 0.3
CUDA_VISIBLE_DEVICES=1 python detect.py --image "/home1/data/congvu/RetinaFace/licenseplate/train/images/newdata1268.jpg" \
    -m /home1/data/dongtrinh/workspace/Retinaface_Ghost/mobilev3_data_v4_modify_negative_2_2_1/mobilev3_Final.pth --network mobilev3 --vis_thres 0.3