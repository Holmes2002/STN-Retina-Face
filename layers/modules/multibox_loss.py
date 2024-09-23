import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
import os
import numpy as np
import cv2
GPU = cfg_mnet['gpu_train']
file_landmark = open('landmark_loss.txt', 'a')
index_image = 0
index_iter = 0
Perspective_mode = True
def invert_image_transform(img, land_pre, land_true ,device='cpu' ):
    # Step 1: Convert tensor back to NumPy array and remove batch dimension
    img = img.squeeze(0).cpu().numpy()

    # Step 2: Transpose the image back to (H, W, C)
    img = img.transpose(1, 2, 0)

    # Step 3: Add the mean values back to the image
    img += (104, 117, 123)

        # Draw circles on each of the corners
    colors = [(0, 0, 255),   # Red
                  (255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 0)]   # Green
    img = np.clip(img, 0, 255).astype(np.uint8).copy() 
    ### Visualize anchors prediction and true
    scale1 = torch.Tensor([680, 680, 680, 680,
                            680, 680,
                           680, 680])
    scale1 = scale1.to(device)
    land_pre = (land_pre * scale1).numpy()
    land_true = (land_true * scale1).numpy()

    land_pre = np.array(land_pre, dtype = np.int32)
    land_true = np.array(land_true, dtype = np.int32)

    for b in [land_true]:
        cv2.circle(img, (b[0], b[1]), 1, (0, 0, 255), 4)
        cv2.circle(img, (b[2], b[3]), 1, (255, 0, 0), 4)
        cv2.circle(img, (b[4], b[5]), 1, (0, 255, 0), 4)
        cv2.circle(img, (b[6], b[7]), 1, (0, 0, 0), 4)
    
    for b in [land_pre]:
        cv2.circle(img, (b[0], b[1]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[2], b[3]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[4], b[5]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[6], b[7]), 1, (255, 0, 255), 4)
    return img



def apply_perspective_transform_to_corners_batch(proposed_boxes, landm_p):
    num_objects = proposed_boxes.shape[0]

    # Calculate the four corners in normalized coordinates for all boxes
    x, y, w, h = proposed_boxes[:, 0], proposed_boxes[:, 1], proposed_boxes[:, 2], proposed_boxes[:, 3]
    x1, y1 = x - w / 2, y - h / 2  # Top-left
    x2, y2 = x + w / 2, y - h / 2  # Top-right
    x3, y3 = x - w / 2, y + h / 2  # Bottom-left
    x4, y4 = x + w / 2, y + h / 2  # Bottom-right
    
    top_left  =  torch.stack([x1 * 2 - 1, y1 * 2 - 1, torch.ones_like(x1)], dim=1)
    top_right =  torch.stack([x2 * 2 - 1, y2 * 2 - 1, torch.ones_like(x1)], dim=1)
    bot_left  =  torch.stack([x3 * 2 - 1, y3 * 2 - 1, torch.ones_like(x1)], dim=1)
    bot_right =  torch.stack([x4 * 2 - 1, y4 * 2 - 1, torch.ones_like(x1)], dim=1)

    # Add the additional value and reshape landm_p to (N, 3, 3)
    affinex = torch.stack([torch.clamp(landm_p[..., 0], min=0.0), landm_p[..., 1], landm_p[..., 2]], dim=-1)
    affiney = torch.stack([landm_p[..., 3], torch.clamp(landm_p[..., 4], min=0.0), landm_p[..., 5]], dim=-1)

    top_left_xy  =  torch.stack([(affinex * top_left).sum(dim=-1), (affiney * top_left).sum(dim=-1)], dim=-1)
    top_right_xy =  torch.stack([(affinex * top_right).sum(dim=-1), (affiney * top_right).sum(dim=-1)], dim=-1)
    bot_left_xy  =  torch.stack([(affinex * bot_left).sum(dim=-1), (affiney * bot_left).sum(dim=-1)], dim=-1)
    bot_right_xy =  torch.stack([(affinex * bot_right).sum(dim=-1), (affiney * bot_right).sum(dim=-1)], dim=-1)

    top_left_xy  =  (top_left_xy + 1)/2
    top_right_xy =  (top_right_xy + 1)/2
    bot_left_xy  =  (bot_left_xy + 1)/2
    bot_right_xy =  (bot_right_xy + 1)/2

    corner_LPs_predict = torch.cat([top_left_xy, top_right_xy, bot_left_xy, bot_right_xy], dim=-1)
    
    return corner_LPs_predict

def apply_perspective_transform_matrix(proposed_boxes, landm_p):
    H, W = 1, 1  # Image dimensions

    # Normalize the source and destination points to the range [-1, 1]
    src_point_normalized = (landm_p / torch.tensor([W, H], dtype=torch.float32, device=landm_p.device)) * 2 - 1
    des_point_normalized = (proposed_boxes / torch.tensor([W, H], dtype=torch.float32, device=proposed_boxes.device)) * 2 - 1

    # Add a column of ones to create homogeneous coordinates
    src_point_augmented = torch.cat([src_point_normalized, torch.ones((src_point_normalized.shape[0], 1), device=src_point_normalized.device)], dim=1)

    # Solve the least squares problem to find the affine transformation matrix A
    # torch.linalg.lstsq is available in PyTorch 1.9+. If you use an earlier version, use torch.solve instead.
    A, _ = torch.linalg.lstsq(src_point_augmented, des_point_normalized)

    # Convert from 3x2 to 2x3 for PyTorch affine transformations (transpose and convert to float32 if needed)
    affine_matrix = A.T.float()
    print(affine_matrix.shape)
    assert False
    return affine_matrix
    # affine_matrix is now differentiable and can be used in subsequent operations

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets, epoch, images):
        global index_image
        global index_iter
        global file_landmark
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        batch_size = loc_data.shape[0]
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 8)
        perspective_matrix_t = torch.Tensor(num, num_priors, 8)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, 12].data
            landms = targets[idx][:, 4:12].data
            perspective_matrix = targets[idx][:, 13:].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx, perspective_matrix, perspective_matrix_t)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            perspective_matrix_t = perspective_matrix_t.cuda()

        zeros = torch.tensor(0).cuda()

        ### landm Loss (Smooth L1)
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        pos_idx2 = pos1.unsqueeze(pos1.dim()).expand_as(landm_t)
        landm_p = landm_data[pos_idx1].view(-1, 6)
        landm_t = landm_t[pos_idx2].view(-1, 8)

        perspective_matrix_t = perspective_matrix_t[pos_idx2].view(-1, 8)
        batch_anchors = priors.data.unsqueeze(0).expand(batch_size, num_priors, 4)
        proposed_boxes = batch_anchors[pos1]       


        ### Visualize batch Anchor
            # batch_anchors = priors.data.unsqueeze(0).expand(1, num_priors, 4)
            # pos1_check = pos1[0,:].unsqueeze(0)
            # proposed_boxes = batch_anchors[pos1_check]      
            # print(proposed_boxes.shape)        
            # invert_anchor_transform(images[0], proposed_boxes)
            # assert False
        ### Visualize Image
        corner_LPs_predict = apply_perspective_transform_matrix(proposed_boxes, landm_p)
        assert False
        corner_LPs_predict = apply_perspective_transform_to_corners_batch(proposed_boxes, landm_p)
        loss_landm = F.smooth_l1_loss(corner_LPs_predict, landm_t, reduction='sum')
        # affine_matrices = compute_affine_matrices(proposed_boxes, landm_t)
        # loss_landm += F.smooth_l1_loss(landm_p, affine_matrices, reduction='sum')
        # Check the correctness of the affine transformation
        
        # loss_landm += 10*F.smooth_l1_loss(corner_LPs_predict_z, torch.ones_like(corner_LPs_predict_z), reduction='sum')
        # loss_landm = torch.sum(torch.sum((corner_LPs_predict - landm_t)**2, dim = 1))
        # loss_landm += torch.sum(torch.sum((corner_LPs_predict_z - torch.ones_like(corner_LPs_predict_z))**2, dim = 1))

        # if index_iter % 50 == 0:
        # # if True:
        #     image = invert_image_transform(images[0], corner_LPs_predict[0].detach().cpu(), landm_t[0].detach().cpu())
        #     os.makedirs('check_images', exist_ok = True)
        #     cv2.imwrite(f"check_images/epoch_{epoch}_{index_image}.jpg", image)
        #     index_image +=1
        
        # file_landmark.write("-"*20 + f'Epoch {epoch}' + "-"*20 + '\n')
        # text = "Affine Predict:\t" + ' '.join([str(i) for i in landm_p.detach().cpu().numpy()[0]]) + "\nAffine True:\t" + ' '.join([str(i) for i in perspective_matrix_t.detach().cpu().numpy()[0]]) 
        # file_landmark.write(text+'\n')
        

        # if index_iter % 1000 == 0:
        #     batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(pos1).cuda()
        #     batch_indices_flat = batch_indices[pos1].view(-1)

        #     for index in range(len(batch_indices_flat)):
        #         index_image = batch_indices_flat[index]
        #         image = invert_image_transform(images[index_image], corner_LPs_predict[index].detach().cpu(), landm_t[index].detach().cpu())
        #         os.makedirs(f'check_images/{epoch}_{index_iter}', exist_ok = True)
        #         cv2.imwrite(f"check_images/{epoch}_{index_iter}/epoch_{epoch}_{index}.jpg", image)
        # index_iter +=1


        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]


        pos = conf_t != zeros
        num_pos = pos.long().sum(1, keepdim=True)

        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        # loss_l -= loss_l
        loss_c /= N
        loss_landm /= N1
        return loss_l, loss_c, loss_landm