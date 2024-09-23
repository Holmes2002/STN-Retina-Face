from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50, cfg_mnetv3, cfg_gnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
import requests
from STN.model import  Advanced_STNet, Residual_STNet, STN_Paddlde
import gradio as gr
from PIL import Image
from ultralytics import YOLO
import os
device = 'cpu'
def resize_with_padding_landmark(image, target_size=(320, 320), fill_color=(0, 0, 0)):
    # Get the original size of the image
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate the new size while maintaining the aspect ratio
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))

    # Resize the image
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    # Create a new image with the target size and fill it with the fill color
    new_image = Image.new("RGB", target_size, fill_color)

    # Calculate paste position to center the resized image
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    # paste_position = (0,0)
    new_image.paste(resized_image, paste_position)


    return new_image

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if True:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = ''
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cpu(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect(img_raw, net, cfg, args):
    img_raw = cv2.resize(img_raw, (680, 680))
    img = np.float32(img_raw)
    print(img.shape)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}ms'.format((time.time() - tic)*1000))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets, img_raw.shape[:2]
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    # show image
    return img_raw
def get_Perspective(label, height, width, input_image, visualize = False):

# Source points from the annotation array
    if visualize:
        height_img, width_img, _ = input_image.shape
        source_points_int = np.int32([
        [label[7]*width/680, label[8]*height/680],   # l0_x, l0_y (top-right)
        [label[5]*width/680, label[6]*height/680],   # l1_x, l1_y (top-left)
        [label[11]*width/680, label[12]*height/680], # l3_x, l3_y (bottom-right)
        [label[13]*width/680, label[14]*height/680], # l3_x, l3_y (bottom-right)
        # [label[16], label[17]]  # l4_x, l4_y (bottom-left)
    ])

        # Draw circles on each of the corners
        colors = [(0, 0, 255),   # Red
                  (255, 0, 0),   # Blue
                  (0, 255, 0),   # Green
                  (0, 0, 0)]     # Black
        # for i, point in enumerate(source_points_int):
        #     cv2.circle(input_image, (point[0], point[1]), radius=5, color=colors[i], thickness=-1)
        #     print((point[0], point[1]), input_image.shape)
        # cv2.imwrite("4_corner.jpg", input_image)
    source_points = np.float32([
        [label[7]*width_img/680, label[8]*height_img/680],   # l0_x, l0_y (top-right)
        [label[5]*width_img/680, label[6]*height_img/680],   # l1_x, l1_y (top-left)
        [label[13]*width_img/680, label[14]*height_img/680], # l3_x, l3_y (bottom-right)
        [label[11]*width_img/680, label[12]*height_img/680], # l3_x, l3_y (bottom-right)
        # [label[16], label[17]]  # l4_x, l4_y (bottom-left)
    ])
    # Define the destination points (e.g., a rectangle)
    # These are the corners of the output image you want to map to
    dest_points = np.float32([
        [width, 0],        # top-right
        [0, 0],            # top-left
        [width, height],   # bottom-right
        [0, height]        # bottom-left
    ])

    # Get the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    if visualize:
        transformed_image = cv2.warpPerspective(input_image, perspective_matrix, (width, height))
        # cv2.imwrite('output_image.jpg', transformed_image)
        # assert False

        return transformed_image
def vehicle_detect(image):
            list_vehicles = []
            model = YOLO("/home1/data/vinhnguyen/Flag_detection/runs_demo_model_s_detect_7_class_v2/train/weights/best.pt")
            results = model([image])

            # Define the class indices to crop (classes 3-6: car, motorcycle, bus, truck)
            target_classes = [3, 4, 5, 6]

            # Process results list
            for idx, result in enumerate(results):
                # Original image (the current frame)
                img = result.orig_img

                # Loop through each detection
                for i, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                    # Convert tensor to numpy array
                    box = box.cpu().numpy().astype(int)
                    cls = int(cls.cpu().numpy())

                    # Check if the class is in target_classes
                    if cls in target_classes:
                        # Crop the detected object using bounding box coordinates
                        x1, y1, x2, y2 = box
                        cropped_img = img[y1:y2, x1:x2]
                        list_vehicles.append(cropped_img)
            return list_vehicles


def init_model():
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='/home1/data/congvu/TMP/Retinaface_Ghost/mobilenet3mobilev3_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobilev3',
                        help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
    parser.add_argument('--image', type=str, default=r'./curve/face.jpg', help='detect images')
    parser.add_argument('--fourcc', type=int, default=1, help='detect on webcam')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_grad_enabled(False)

    if args.network == "mobile0.25":
        from models.retinaface_m import RetinaFace
        cfg = cfg_mnet
    elif args.network == "resnet50":
        from models.retinaface_m import RetinaFace
        cfg = cfg_re50
    elif args.network == "ghostnet":
        from models.retinaface_g import RetinaFace
        cfg = cfg_gnet
    elif args.network == "mobilev3":
        from models.retinaface_g import RetinaFace
        cfg = cfg_mnetv3

    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cpu")
    net = net.to(device)
    return net, cfg, args


def draw_img(image):
    model_Detect, cfg, args = init_model()
    model_STN = Advanced_STNet()
    model_STN.load_state_dict(torch.load('/home/data2/congvu/checkpoint/30_94.pt'))
    model_STN.to('cpu')
    model_STN.eval()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    list_vehicles = vehicle_detect(image)
    vis_list = []
    if len(list_vehicles) == 0:
        list_vehicles = [image]
    for image in list_vehicles:
        dets, size = detect(image, model_Detect, cfg, args)
        height, width, _ = image.shape
        scale_w = width / size[1]
        scale_h = height / size[0]
        
        list_crop_lp = []
        for idx, b in enumerate(dets):
            if b[4] < 0.5: continue   
            x1 = int(b[0] * scale_w)
            y1 = int(b[1] * scale_h)
            x2 = int(b[2] * scale_w)
            y2 = int(b[3] * scale_h)
            
            # Clamp coordinates to the image boundaries
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

    # Crop the image
            cropped_lp = image[y1:y2, x1:x2]
            list_crop_lp.append(cropped_lp)
        if len(list_crop_lp) == 0:
            list_crop_lp = [image]
        for cropped_lp in list_crop_lp:
            # box = [int(b[0] * scale_w), int(b[1] * scale_h), int(b[2] * scale_w), int(b[3] * scale_h)] 
            # cropped_lp = image[box[1]:box[3], box[0]:box[2]]
            resize_lp = resize_with_padding_landmark(Image.fromarray(cropped_lp)).convert("L")
            resize_lp_np = np.array(resize_lp, dtype = np.uint8)
            resize_lp = torch.tensor(np.array(resize_lp)/255.).unsqueeze(0).unsqueeze(0).float().cpu()

            preidct_lp, _ = model_STN(resize_lp)
            preidct_lp = preidct_lp.detach().cpu().numpy()[0][0]
            preidct_lp = np.array(preidct_lp*255, dtype = np.uint8)
            vis = np.concatenate((resize_lp_np, preidct_lp), axis=1)
            vis_list.append(vis)
    print("inference completed")
    if len(vis_list) == 0:
            return None
    vis_final = np.concatenate(vis_list, axis=0)
    output_image = cv2.cvtColor(vis_final, cv2.COLOR_GRAY2RGB)
    os.makedirs('gradio_result', exist_ok = True)
    cv2.imwrite('gradio_result/result.jpg', output_image)
    return Image.fromarray(output_image)

demo = gr.Interface(
    draw_img, 
    inputs=gr.Image(),  # Image input
    outputs=gr.Image(),  # Image output
    title="STN License Plate",
    # examples=[glob("./test_image/*.png")[:3]]
    examples= []
)

demo.launch(show_tips=True, server_name='10.9.3.239', server_port=3190,share=True,debug=True)
