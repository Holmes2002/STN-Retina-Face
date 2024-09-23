docker run -it --gpus device=0       \
     -v /home1/data/congvu/RetinaFace:/yolov4               \
    nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5               \
    yolo_v4 kmeans -l /yolov4/Reanchors -i /yolov4/licenseplate/train/images -x 680 -y 680 -n 3