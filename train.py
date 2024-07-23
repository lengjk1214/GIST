import argparse
from train_nano import train_nano
from train_breast import train_breast
from train_breast2 import train_breast2
from train_colorectal import train_colorectal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nanostring', help='should be nanostring, breast, breast2, colorectal')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='../datasets/nanostring/nanostring_Lung13')
    parser.add_argument('--epochs', type=int, default=1000) 
    parser.add_argument('--id', type=str, default='fov1')
    parser.add_argument('--img_name', type=str, default='F001')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/lung13')
    parser.add_argument('--result_path', type=str, default='../results/lung13')
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='final_0.pth')
    parser.add_argument('--cluster_method', type=str, default='leiden', help='leiden or mclust')
 
    opt = parser.parse_args() 
    if opt.dataset == 'nanostring':
        train_nano(opt)

    elif opt.dataset == 'breast':
        train_breast(opt)

    elif opt.dataset == 'breast2':
        train_breast2(opt)
 
    elif opt.dataset == 'colorectal':
        train_colorectal(opt)


