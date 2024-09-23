import cv2
import torch
import numpy as np
from tqdm import tqdm
from STN.dataset import *
from torchvision import transforms
import os
import argparse
from utils import utils
import torch
from torchvision import transforms as T
from STN.model import  Advanced_STNet, Residual_STNet, STN_Paddlde
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast, GradScaler

des = '/home/data2/congvu'
PIXEL_LOSS_WEIGHT = 10
THETHA_LOSS_WEIGHT = 200
PERCEP_LOSS_WEIGHT = 20
scaler = GradScaler(growth_interval=100)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default='laion2b_s34b_b88k',
        help="The pretrained weight of model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--freeze",type = str,
        default=False,
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--float16",
        default=False,
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args

def convert_image_np(image):
    img = image.detach().cpu().numpy()
    denormalized_image = np.array(img*255, dtype = np.int32)
    return np.array(denormalized_image[0])

# def convert_image_np(image):

#     img=image.detach().cpu()
#     denormalize = transforms.Normalize(mean=[0.], std=[1.])
#     denormalized_tensor = denormalize(image)
#     denormalized_image = transforms.ToPILImage()(denormalized_tensor)
#     return np.array(denormalized_image)

def evaluate(model, val_loader, epoch, iteration, train_loader, type = 'val'):
    model.eval()
    iter_val = iter(val_loader)
    print("  - Start Evaluate")
    test_loss = 0
    correct = 0
    idx = 0
    iter_val = iter(val_loader)

    STN_loss = 0
    for data,target,theta in iter_val:
            correct += 1
            data,target, theta = data.to(device),target.to(device), theta.to(device)
            output, theta_predict = model(data)
            STN_loss += nn.MSELoss()(theta_predict, theta).detach().cpu()
            for i in range(output.shape[0]):
                    idx +=1
                    output_ = convert_image_np(output[i])
                    target_ = convert_image_np(target[i])
                    data_ = convert_image_np(data[i])
                    # output_ = output_[:,:,::-1]
                    # target_ = target_[:,:,::-1]
                    # data_ = data_[:,:,::-1]
                    vis = np.concatenate((data_, output_, target_), axis=1)
                    os.makedirs(f"{des}/results/{type}/{epoch}/{iteration}",exist_ok=True)
                    cv2.imwrite(f"{des}/results/{type}/{epoch}/{iteration}/{idx}.png",vis)

    iter_val = iter(train_loader)
    for data,target,_ in iter_val:
            data,target =data.to(device),target.to(device)
            output, _ = model(data)
            for i in range(output.shape[0]):
                    idx +=1
                    output_ = convert_image_np(output[i])
                    target_ = convert_image_np(target[i])
                    data_ = convert_image_np(data[i])
                    # output_ = output_[:,:,::-1]
                    # target_ = target_[:,:,::-1]
                    # data_ = data_[:,:,::-1]
                    vis = np.concatenate((data_, output_, target_), axis=1)
                    os.makedirs(f"{des}/results/train/{epoch}/{iteration}",exist_ok=True)
                    cv2.imwrite(f"{des}/results/train/{epoch}/{iteration}/{idx}.png",vis)

    print("  - End Eval")
    model.train()
    return STN_loss/correct

device = torch.device('cuda')
def main(args):
    
    ### Dataset
    train_transform_aug = transforms.Compose([
    # transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),transforms.Normalize(mean=[0.], std=[1.])])
    train_loader = get_dataloader_train(args.root, (320, 320), train_transform_aug,args.batch_size, True)
    num_batches = len(train_loader)
    val_loader = get_dataloader_train(args.root, (320, 320),  train_transform_aug,2, training = False)
    sub_train_loader = get_dataloader_train(args.root, (320, 320),  train_transform_aug,2, training = True, visualize_train = True)
    # test_loader = get_dataloader_train('test_real.txt', "/home/data/congvu/Face_Align/test_img", train_transform_target, train_transform_target,4,True, False)

    ### Model
    model = Advanced_STNet()
    # model.load_state_dict(torch.load('checkpoint/71_26.pt'))
    model.to(device)
    params      = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = utils.cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    print(f'  - Init  with cosine learning rate scheduler {args.lr}')
    theta_l2 = nn.MSELoss()

    best_STN = 1e9
    # loss_fn = VGGPerceptualLoss().to('cuda')
    for epoch in range(0,args.epochs):

        model.train()
        count = 0
        iter_train = iter(train_loader)
        # tmp_stn = evaluate(model, val_loader, epoch, count, sub_train_loader)
        for i in tqdm(train_loader):
            step = count + epoch * num_batches
            count+=1
            lr = scheduler(step)
            optimizer.zero_grad()
            image_aug, image, theta = next(iter_train)
            image_aug, image, theta = image_aug.to(device), image.to(device), theta.to(device)
            if args.float16:
                predict_image, predict_theta = model(image_aug)
                # predict_image, predict_theta = model(image_aug)
                loss_percep  = 0
                loss_pixel = torch.mean(torch.abs(predict_image - image))
                
                # predict_theta[:,0,1] = -predict_theta[:,0,1]
                # predict_theta[:,1,0] = -predict_theta[:,1,0]

                theta_loss = theta_l2(predict_theta, theta)
                loss = loss_percep*PERCEP_LOSS_WEIGHT + loss_pixel*PIXEL_LOSS_WEIGHT + theta_loss*THETHA_LOSS_WEIGHT
                # loss = loss_percep*PERCEP_LOSS_WEIGHT + loss_pixel*PIXEL_LOSS_WEIGHT 
                scaler.scale(loss).backward()
                # Unscales gradients and performs optimizer step
                scaler.step(optimizer)
                scaler.update()

            else:
                predict_image, predict_theta = model(image_aug)
                # predict_image, predict_theta = model(image_aug)
                loss_percep  = 0
                loss_pixel = torch.mean(torch.abs(predict_image - image))
                
                # predict_theta[:,0,1] = -predict_theta[:,0,1]
                # predict_theta[:,1,0] = -predict_theta[:,1,0]

                theta_loss = theta_l2(predict_theta, theta)
                loss = loss_percep*PERCEP_LOSS_WEIGHT + loss_pixel*PIXEL_LOSS_WEIGHT + theta_loss*THETHA_LOSS_WEIGHT
                # loss = loss_percep*PERCEP_LOSS_WEIGHT + loss_pixel*PIXEL_LOSS_WEIGHT 
                loss.backward()
                optimizer.step()

            print(f"EPOCH {epoch}, Loss {loss} Per {loss_percep} Pixel {loss_pixel*PIXEL_LOSS_WEIGHT} Theta {theta_loss*THETHA_LOSS_WEIGHT} best_STN {best_STN}")
        tmp_stn = evaluate(model, val_loader, epoch, count, sub_train_loader)

                # evaluate(model, test_loader, epoch, count, 'test')
        os.makedirs(f'{des}/checkpoint', exist_ok = True)
        torch.save(model.state_dict(), f'{des}/checkpoint/{epoch}_{count}.pt')
        if tmp_stn < best_STN:
            torch.save(model.state_dict(), f'{des}/checkpoint/best.pt')
            best_STN = tmp_stn
if __name__ == '__main__':
    args = parse_arguments()
    main(args)