import argparse
import os
import cv2
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import evaluate
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image


class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image


class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()     
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res



def load_gt(root, cls):
    gt = []
    gt_dir = os.path.join(root, cls, 'ground_truth')
    sub_dirs = sorted(os.listdir(gt_dir))
    for sb in sub_dirs:
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
            gt.append(temp)
    gt = np.concatenate(gt, 0)
    return gt


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    # required training super-parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--category", type=str , default='leather', help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')

    parser.add_argument("--checkpoint-epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch-size", type=int, default=32, help='batch size')
    # trivial parameters
    parser.add_argument("--result-path", type=str, default='results', help="save results")
    parser.add_argument("--save-fig", action='store_true', help="save images with anomaly score")
    parser.add_argument("--mvtec-ad", type=str, default='mvtec_anomaly_detection', help="MvTec-AD dataset path")
    parser.add_argument('--model-save-path', type=str, default='snapshots', help='path where student models are saved')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.png')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.split == 'test':
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
        test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
        test_pos_image_list = sorted(list(test_pos_image_list))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
    teacher.cuda()
    student.cuda()

    if args.split == 'train':
        train_val(teacher, student, train_loader, val_loader, args)
    elif args.split == 'test':
        saved_dict = torch.load(args.checkpoint)
        category = args.category
        gt = load_gt(args.mvtec_ad, category)

        print('load ' + args.checkpoint)
        student.load_state_dict(saved_dict['state_dict'])

        pos = test(teacher, student, test_pos_loader)
        neg = test(teacher, student, test_neg_loader)

        scores = []
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (256, 256))
            scores.append(temp)
        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (256, 256))
            scores.append(temp)

        scores = np.stack(scores)
        neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)
        gt_pixel = np.concatenate((gt, neg_gt), 0)
        gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)        

        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
     


def test(teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        batch_img = batch_img.cuda()
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_img.size(0)
    return loss_map
    

def train_val(teacher, student, train_loader, val_loader, args):
    min_err = 10000
    teacher.eval()
    student.train()

    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        student.train()
        for batch_data in train_loader:
            _, batch_img = batch_data
            batch_img = batch_img.cuda()

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss =  0
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        err = test(teacher, student, val_loader).mean()
        print('Valid Loss: {:.7f}'.format(err.item()))
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.model_save_path, args.category, 'best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': args.category,
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)

if __name__ == "__main__":
    main()
