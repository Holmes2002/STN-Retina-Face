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
import os 
import random
def apply_perspective_transform_to_corners_batch(proposed_boxes, landm_p):
    num_objects = proposed_boxes.shape[0]

    # Calculate the four corners in normalized coordinates for all boxes
    x, y, w, h = proposed_boxes[:, 0], proposed_boxes[:, 1], proposed_boxes[:, 2], proposed_boxes[:, 3]
    x1, y1 = x - w / 2, y - h / 2  # Top-left
    x2, y2 = x + w / 2, y - h / 2  # Top-right
    x3, y3 = x - w / 2, y + h / 2  # Bottom-left
    x4, y4 = x + w / 2, y + h / 2  # Bottom-right

    # Stack corners into a (N, 4, 3) tensor with homogeneous coordinates
    # corners = torch.stack([
    #     torch.stack([x1, y1, torch.ones_like(x1)], dim=1),
    #     torch.stack([x2, y2, torch.ones_like(x1)], dim=1),
    #     torch.stack([x3, y3, torch.ones_like(x1)], dim=1),
    #     torch.stack([x4, y4, torch.ones_like(x1)], dim=1)
    # ], dim=1)  # Resulting shape: (N, 4, 3)
    
    top_left  =  torch.stack([x1, y1, torch.ones_like(x1)], dim=1)
    top_right =  torch.stack([x2, y2, torch.ones_like(x1)], dim=1)
    bot_left  =  torch.stack([x3, y3, torch.ones_like(x1)], dim=1)
    bot_right =  torch.stack([x4, y4, torch.ones_like(x1)], dim=1)

    # Add the additional value and reshape landm_p to (N, 3, 3)
    affinex = torch.stack([torch.clamp(landm_p[..., 0], min=0.0), landm_p[..., 1], landm_p[..., 2]], dim=-1)
    affiney = torch.stack([landm_p[..., 3], torch.clamp(landm_p[..., 4], min=0.0), landm_p[..., 5]], dim=-1)

    top_left_xy  =  torch.stack([(affinex * top_left).sum(dim=-1), (affiney * top_left).sum(dim=-1)], dim=-1)
    top_right_xy =  torch.stack([(affinex * top_right).sum(dim=-1), (affiney * top_right).sum(dim=-1)], dim=-1)
    bot_left_xy  =  torch.stack([(affinex * bot_left).sum(dim=-1), (affiney * bot_left).sum(dim=-1)], dim=-1)
    bot_right_xy =  torch.stack([(affinex * bot_right).sum(dim=-1), (affiney * bot_right).sum(dim=-1)], dim=-1)
    corner_LPs_predict = torch.cat([top_left_xy, top_right_xy, bot_left_xy, bot_right_xy], dim=-1)
    
    return corner_LPs_predict

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
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect(img_raw):
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
    anchors_data = priors.data.squeeze(0).cpu().numpy()
    landms = apply_perspective_transform_to_corners_batch(prior_data, landms.data.squeeze(0) )
    landms = decode_landm(landms, prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2],
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
    anchors_data = anchors_data[inds]
    for index, b in enumerate(landms):
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        # landms
        cv2.circle(img_raw, (b[0], b[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[2], b[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[4], b[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[6], b[7]), 1, (0, 255, 0), 4)
    return img_raw

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    anchors_data = anchors_data[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]
    anchors_data = anchors_data[keep]
    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]
    anchors_data = anchors_data[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    for index, b in enumerate(dets):
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
        # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        ### anchors
        # x,y,w,h = [i*680 for i in anchors_data[index,:]]
        # top_left_x = int(x - w / 2)
        # top_left_y = int(y - h / 2)

        # # Calculate bottom-right corner
        # bottom_right_x = int(x + w / 2)
        # bottom_right_y = int(y + h / 2)

        # # Draw the rectangle (color: white, thickness: 2)
        # cv2.rectangle(img_raw, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)

    # show image
    return img_raw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50',
                        help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
    parser.add_argument('--image', type=str, default=r'./curve/face.jpg', help='detect images')
    parser.add_argument('--fourcc', type=int, default=1, help='detect on webcam')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
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
    device = 'cpu'
    net = net.to(device)

    if args.fourcc ==0:
        cap = cv2.VideoCapture(0)
        cap.set(3, 720)  # set video width
        cap.set(4, 680)  # set video height

        while True:
            ret, frame = cap.read()
            # img = cv2.imread(frame)
            img_raw = detect(frame)
            cv2.imshow('fourcc', frame)
            k = cv2.waitKey(20)
            # q键退出
            if (k & 0xff == ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        root = "/home1/data/congvu/RetinaFace/licenseplate/train/images"
        list_image = os.listdir(root)
        # random.shuffle(list_image)
        list_image = list_image[:40]
        for index, file in enumerate(list_image):
            img = cv2.imread(f"{root}/{file}")
            img = cv2.resize(img, (680, 680))
            img_raw = detect(img)

            # save image
            if args.save_image:
                os.makedirs('output_folder_only_landmark', exist_ok = True)
                name = f'output_folder_only_landmark/out_{index}' + '.jpg'
                cv2.imwrite(name, img_raw)




