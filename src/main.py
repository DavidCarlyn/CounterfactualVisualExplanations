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

def visualize_feature(model, img, img_features, img2, img2_features, idx, idx2):
    col = idx // img_features.shape[1]
    row = idx % img_features.shape[1]
    inv_size1 = get_size_before_conv(img_features.shape[1], model.maxpool)
    window = [row*2, (row*2)+1, col*2, (col*2)+1]
    inv_size2 = get_size_before_conv(inv_size1, model.conv2)
    window = [window[0]*2, (window[1]*2)+2, window[2]*2, (window[3]*2)+2]
    img_size = get_size_before_conv(inv_size2, model.conv1) + 1
    window = [window[0]*2, (window[1]*2)+2, window[2]*2, (window[3]*2)+2]

    col = idx // img2_features.shape[1]
    row = idx % img2_features.shape[1]
    inv_size1 = get_size_before_conv(img2_features.shape[1], model.maxpool)
    window2 = [row*2, (row*2)+1, col*2, (col*2)+1]
    inv_size2 = get_size_before_conv(inv_size1, model.conv2)
    window2 = [window2[0]*2, (window2[1]*2)+2, window2[2]*2, (window2[3]*2)+2]
    img_size = get_size_before_conv(inv_size2, model.conv1) + 1
    window2 = [window2[0]*2, (window2[1]*2)+2, window2[2]*2, (window2[3]*2)+2]
    img[0, window[0]:window[1], window[2]:window[3]] = img2[0, window2[0]:window2[1], window2[2]:window2[3]]
    ToPILImage()(img).save("test.png")

    with torch.no_grad():
        img = img.cuda()
        _, out = model(img.unsqueeze(0))
        sm = nn.Softmax()
        print(sm(out[0]))

    exit()

def main():
    args = get_args()
    train_dset, test_dset = get_datasets()
    model = train(train_dset, test_dset, args)
    one_img = None
    four_img = None
    for img, lbl in test_dset:
        if lbl == 1:
            one_img = img
        
        if lbl == 4:
            four_img = img

        if one_img is not None and four_img is not None:
            break

    ToPILImage()(one_img).save("one.png")
    ToPILImage()(four_img).save("four.png")
    
    with torch.no_grad():
        S = []
        one_features = model.extract_features(one_img.unsqueeze(0).cuda())[0]
        four_features = model.extract_features(four_img.unsqueeze(0).cuda())[0]
        visualize_feature(model, one_img, one_features, four_img, four_features, 5, 4)
        one_features = one_features.view(-1, one_features.shape[1] * one_features.shape[2])
        four_features = four_features.view(-1, four_features.shape[1] * four_features.shape[2])

        sm = nn.Softmax()
        max_loops = one_features.shape[1]**2
        for _ in range(max_loops):
            edit, conf = get_next_edit(one_features, four_features, 4, model, S)
            print(edit)
            S.append(edit)
            #out = model.predict(one_features.unsqueeze(0))[0]
            one_features[:, edit[0]] = four_features[:, edit[1]]
            out = model.predict(one_features.unsqueeze(0))[0]
            conf = sm(out)
            if conf[4] == max(conf): break
        
        print(f"Number of edits: {len(S)}")
        print(f"Confidence: {conf[4]}")


    



if __name__ == "__main__":
    main()