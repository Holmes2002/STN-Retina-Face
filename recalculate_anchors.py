import tqdm
import torch, cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.ops import box_iou
image_paths = np.loadtxt("images.txt", dtype=str) # all image paths
bboxes_file_path = "boxes.txt"
bboxes = torch.tensor(np.loadtxt(bboxes_file_path, dtype=int, delimiter=" ", ndmin=2)) # Take only x, y, w, h 
bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] # get corner point
bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] # get corner point


def scale_bbox(bbox, image, o_shape):
    W, H, C = o_shape
    w, h, c = image.shape
    Wratio = w/W
    Hratio = h/H
    ratioList = [Hratio, Wratio, Hratio, Wratio]
    bbox = [int(a * b) for a, b in zip(bbox, ratioList)]
    return bbox

# Function that returns redefined bbox
def function(image, bboxs, base=512):
    H, W, C = image.shape
    if H > W:
        height_persentage = float(base/H)
        width_size = int(W*height_persentage)
        resized_image = cv2.resize(image, (width_size, base), interpolation=cv2.INTER_CUBIC)
        h, w, c = resized_image.shape
        bbox = scale_bbox(bboxs, resized_image, (H, W, C))
        width1 = (base - w) // 2
        width2 = (base - w) - width1
        bbox = [bbox[0]+width1, bbox[1], bbox[2]+width2, bbox[3]]
        
        # Symmetric Padding
        mask = np.array(np.zeros(shape=(base, width1, C)), dtype=int)
        resized_image = np.concatenate((resized_image, mask), axis=1)

        mask = np.array(np.zeros(shape=(base, width2, C)), dtype=int)
        resized_image = np.concatenate((mask, resized_image), axis=1)
        # display(resized_image, bbox)
        
    else:
        width_percentage = float(base/W)
        height_size = int(H*width_percentage)
        resized_image = cv2.resize(image, (base, height_size), interpolation=cv2.INTER_CUBIC)
        h, w, c = resized_image.shape
        bbox = scale_bbox(bboxs, resized_image, (H, W, C))
        height1 = (base - h) // 2
        height2 = (base - h) - height1
        bbox = [bbox[0], bbox[1]+height1, bbox[2], bbox[3]+height2]
        
        # Symmetric Padding
        mask = np.array(np.zeros(shape=(height1, base, C)), dtype=int)
        resized_image = np.concatenate((resized_image, mask))
        
        mask = np.array(np.zeros(shape=(height2, base, C)), dtype=int)
        resized_image = np.concatenate((mask, resized_image))
        # display(resized_image, bbox)
    
    return bbox


redefined_bboxes = []

for idx in tqdm.trange(len(image_paths)):
    image = np.array(Image.open(image_paths[idx]).convert("RGB"))
    bbox = bboxes[idx].numpy()
    bbox = function(image, bbox)
    redefined_bboxes.append(bbox)
redefined_bboxes = torch.tensor(redefined_bboxes, dtype=torch.float)

def IoU(clusters: torch.tensor, bboxes: torch.tensor):
    iou_values = box_iou(clusters, bboxes)
    return iou_values
def KMeans(bboxes:torch.tensor, k:int, dist=torch.mean, stop_iter=5):
    rows = bboxes.shape[0]
    distances = torch.empty((rows, k))
    last_clusters = torch.zeros((rows, ))

    cluster_indxs = np.random.choice(rows, k, replace=False) # choose unique indexs in rows
    clusters = bboxes[cluster_indxs].clone()

    iteration = 0
    while True:
        # calculate the distances 
        distances = IoU(bboxes, clusters)

        nearest_clusters = torch.argmax(distances, dim=1) # 0, 1, 2 ... K   

        if (last_clusters == nearest_clusters).all(): # break if nothing changes
            iteration += 1
            if iteration == stop_iter:
                break
        else:
            iteration = 0
        # Take the mean and step for cluster coordiantes 
        for cluster in range(k):
            clusters[cluster] = torch.mean(bboxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters.clone()
    return clusters, distances
clusterList = [1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12]
mean_distances, anchorBoxes = [], []

for cluster_k in tqdm.tqdm(clusterList):
    anchors, distances = KMeans(redefined_bboxes, k=cluster_k)
    indxs = torch.argmax(distances, dim=1)
    filtered_distances = []
    for i, distance in enumerate(distances):
        filtered_distances.append(distance[indxs[i]].item())
    mean_distances.append(np.mean(filtered_distances))
    anchorBoxes.append(anchors)

for anchor in anchors:
	print(anchor)
plt.plot(clusterList, mean_distances)
plt.scatter(clusterList, mean_distances)
plt.title("Mean IoU Score")
plt.xlabel("Number of clusters")
plt.ylabel("IoU score")
plt.savefig("mean_iou_score.png", dpi=300, bbox_inches='tight')
