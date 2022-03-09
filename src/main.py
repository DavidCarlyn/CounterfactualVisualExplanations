import random
import math

from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import MNIST_CNN

def get_size_after_conv(cur_size, module):
    stride = module.stride
    kernel_size = module.kernel_size
    padding = module.padding
    if not isinstance(stride, int):
        stride = stride[0]
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    if not isinstance(padding, int):
        padding = padding[0]
    
    return ((cur_size + padding*2 - kernel_size) / stride) + 1

def get_size_before_conv(cur_size, module):
    stride = module.stride
    kernel_size = module.kernel_size
    padding = module.padding
    if not isinstance(stride, int):
        stride = stride[0]
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    if not isinstance(padding, int):
        padding = padding[0]

    return stride * (cur_size - 1) + kernel_size - padding*2

def get_datasets():
    train_dset = datasets.MNIST(
        root='../datasets',
        train=True,
        transform=ToTensor(),
        download=True
    )
    test_dset = datasets.MNIST(
        root='../datasets',
        train=False,
        transform=ToTensor(),
        download=True
    )

    return train_dset, test_dset

def train(train_dset, test_dset, args):
    model = MNIST_CNN().cuda()
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        return model

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_dset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dset, batch_size=16, shuffle=False, num_workers=4)
    for epoch in range(args.epochs):
        total_loss = 0
        for imgs, lbls in tqdm(train_dataloader):
            x, y = model(imgs.cuda())
            loss = loss_fn(y, lbls.cuda())
            total_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: {total_loss}")

        with torch.no_grad():
            correct = 0
            total = 0
            for imgs, lbls in tqdm(test_dataloader):
                x, y = model(imgs.cuda())
                for pred, lbl in zip(y, lbls):
                    pred_c = torch.argmax(pred)
                    total += 1
                    if pred_c == lbl.item():
                        correct += 1
            print(f"Epoch {epoch}: {round(correct/total, 4)*100}%")

    torch.save(model.state_dict(), "model.pt")
    return model

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--load", type=str, default=None)
    
    args = parser.parse_args()
    return args

def get_next_edit(feat_original, feat_target, tgt_lbl, model, S=[]):
    highest_conf = 0.0
    edit = [-1, -1]
    sm = nn.Softmax()
    with torch.no_grad():
        for i in range(feat_original.shape[1]):
            if len(list(filter(lambda x: x[0] == i, S))): continue
            for j in range(feat_target.shape[1]):
                if len(list(filter(lambda x: x[1] == j, S))): continue
                tmp = feat_original[:, i].clone()
                feat_original[:, i] = feat_target[:, j]
                out = model.predict(feat_original.unsqueeze(0))[0]
                conf = sm(out)
                if conf[tgt_lbl] > highest_conf:
                    highest_conf = conf[tgt_lbl]
                    edit = [i, j]
                feat_original[:, i] = tmp
    return edit, highest_conf

def visualize_feature(model, img, img_features, img2, img2_features, S):
    for idx, idx2 in S:
        col = idx % img_features.shape[1]
        row = idx // img_features.shape[1]
        inv_size1 = get_size_before_conv(img_features.shape[1], model.maxpool)
        window = [row*2, (row*2)+1, col*2, (col*2)+1]
        inv_size2 = get_size_before_conv(inv_size1, model.conv2)
        window = [window[0]*2, (window[1]*2)+2, window[2]*2, (window[3]*2)+2]
        img_size = get_size_before_conv(inv_size2, model.conv1) + 1
        window = [window[0]*2, (window[1]*2)+2, window[2]*2, (window[3]*2)+2]

        col = idx2 % img2_features.shape[1]
        row = idx2 // img2_features.shape[1]
        inv_size1 = get_size_before_conv(img2_features.shape[1], model.maxpool)
        window2 = [row*2, (row*2)+1, col*2, (col*2)+1]
        inv_size2 = get_size_before_conv(inv_size1, model.conv2)
        window2 = [window2[0]*2, (window2[1]*2)+2, window2[2]*2, (window2[3]*2)+2]
        img_size = get_size_before_conv(inv_size2, model.conv1) + 1
        window2 = [window2[0]*2, (window2[1]*2)+2, window2[2]*2, (window2[3]*2)+2]
        img[0, window[0]:window[1], window[2]:window[3]] = img2[0, window2[0]:window2[1], window2[2]:window2[3]]
    ToPILImage()(img).save("composite.png")

    with torch.no_grad():
        img = img.cuda()
        _, out = model(img.unsqueeze(0))
        sm = nn.Softmax()
        print(sm(out[0]))

def main():
    args = get_args()
    train_dset, test_dset = get_datasets()
    model = train(train_dset, test_dset, args)
    source_lbl = random.choice(range(10))
    tmp = list(range(10))
    tmp.pop(source_lbl)
    distractor_lbl = random.choice(tmp)
    source_img = None
    distractor_img = None
    for img, lbl in test_dset:
        if lbl == source_lbl:
            source_img = img
        
        if lbl == distractor_lbl:
            distractor_img = img

        if source_img is not None and distractor_img is not None:
            break

    ToPILImage()(source_img).save("source.png")
    ToPILImage()(distractor_img).save("distractor.png")
    
    with torch.no_grad():
        S = []
        source_features = model.extract_features(source_img.unsqueeze(0).cuda())[0]
        distractor_features = model.extract_features(distractor_img.unsqueeze(0).cuda())[0]
        source_features = source_features.view(-1, source_features.shape[1] * source_features.shape[2])
        distractor_features = distractor_features.view(-1, distractor_features.shape[1] * distractor_features.shape[2])

        sm = nn.Softmax()
        max_loops = source_features.shape[1]**2
        for _ in range(max_loops):
            edit, conf = get_next_edit(source_features, distractor_features, distractor_lbl, model, S)
            print(edit)
            S.append(edit)
            #out = model.predict(source_features.unsqueeze(0))[0]
            source_features[:, edit[0]] = distractor_features[:, edit[1]]
            out = model.predict(source_features.unsqueeze(0))[0]
            conf = sm(out)
            if conf[distractor_lbl] == max(conf): break
        
        print(f"Number of edits: {len(S)}")
        print(f"Confidence: {conf[distractor_lbl]}")

        visualize_feature(model, 
            source_img, 
            source_features.view(source_features.shape[0], int(math.sqrt(source_features.shape[1])), -1), 
            distractor_img, 
            distractor_features.view(distractor_features.shape[0], int(math.sqrt(distractor_features.shape[1])), -1),
            S)

    



if __name__ == "__main__":
    main()