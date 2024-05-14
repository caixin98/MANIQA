import argparse
import os
import torch
import numpy as np
import random
import cv2
from torchvision import transforms
from models.maniqa import MANIQA
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from glob import glob
from utils.inference_process import ToTensor, Normalize


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform, num_crops=20):
        super(ImageFolder, self).__init__()
        self.folder_path = folder_path
        self.image_paths = glob(os.path.join(folder_path, "*.png")) + glob(os.path.join(folder_path, "*.jpg"))
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img_name = image_path.split('/')[-1]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype('float32') / 255

        h, w, c = img.shape
        
        new_h, new_w = 224, 224
        if h < new_h or w < new_w:
            #resize and keep the aspect ratio
            if h < w:
                resize_h = 224
                resize_w = int(w * 224 / h)
            else:
                resize_w = 224
                resize_h = int(h * 224 / w)
            img = cv2.resize(img, (resize_w + 1, resize_h + 1))
            # print(f"Resized image {img_name} to {img.shape}")
            h, w = img.shape[0], img.shape[1]
        
        img = np.transpose(img, (2, 0, 1))
        img_patches = []
        for _ in range(self.num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = img[:, top: top + new_h, left: left + new_w]
            img_patches.append(patch)

        img_patches = np.array(img_patches)
        
        return img_name, img_patches

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict scores for all images in a folder")
    parser.add_argument('folder_path', type=str, help='Path to the image folder')
    args = parser.parse_args()

    setup_seed(20)
    
    num_crops = 20
    transform = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
    dataset = ImageFolder(args.folder_path, transform, num_crops=num_crops)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model definition and configuration as before
    # net = setup_model() (Assuming setup_model is a placeholder function; fill in accordingly)

    config = Config({
        # image path
        # "image_path": "./test_images/kunkun.png",

        # valid times
        "num_crops": 20,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # checkpoint path
        "ckpt_path": "/root/caixin/StableSR/MANIQA/ckpt_koniq10k.pt",
    })
    
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()
    net.eval()

    total_score = 0
    count = 0

    for img_name, img_patches in tqdm(dataloader):
        img_patches = img_patches.squeeze(0)
        scores = 0

        for i in range(num_crops):
            patch = img_patches[i].cuda()
            patch = patch.unsqueeze(0)
            score = net(patch)
            scores += score.item()
        
        avg_score = scores / num_crops
        total_score += avg_score
        count += 1
        # print(f"Image {img_name[0]} score: {avg_score}")

    if count > 0:
        print("Average score of all images:", total_score / count)