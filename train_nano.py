import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match, seed_everything
from train_transformer import train_nano_fov, test_nano_fov
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import pickle
import timm
from ctran import ctranspath


def gen_adatas(root, id, img_name, model):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ncluster = len(set(adata.obs['merge_cell_type']))

    print(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s.jpg'%(img_name)))
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.ToTensor()
    img = transform(img)

    df = pd.DataFrame(index=adata.obs.index)
    df['cx'] = adata.obs['cx']
    df['cy'] = adata.obs['cy']
    arr = df.to_numpy()
    adata.obsm['spatial'] = arr

    patchs = []
    w, h = 120, 120

   
    for (x, y) in zip(adata.obs['cx'], adata.obs['cy']):
        
        if x < w or y < h:
            img_p = img[:, int(y-60):int(y+60), int(x-60): int(x+60)]
        else:
            img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]
        resize = transforms.Resize((224, 224))
        image_resized = resize(img_p)
        image_resized = image_resized.unsqueeze(0)
        with torch.no_grad():
            feature = model(image_resized)
            feature = feature.cpu().numpy()
        feature = torch.tensor(feature.squeeze())
        patchs.append(feature.squeeze())
        

    patchs = np.stack(patchs)
    df = pd.DataFrame(patchs, index=adata.obs.index)
    adata.obsm['imgs'] = df

    Cal_Spatial_Net(adata, rad_cutoff=80)
    Stats_Spatial_Net(adata)
    return adata


@torch.no_grad()
def infer(opt, r=0):
    seed = opt.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #load model
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'../checkpoint/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()


    sp = os.path.join(opt.save_path)
    file_path = os.path.join(sp, "adatas_list.pkl")
    with open(file_path, 'rb') as file:
        adatas = pickle.load(file)
        
    pth_list = ['final_200_0.pth','final_300_0.pth','final_400_0.pth',
            'final_500_0.pth','final_600_0.pth','final_700_0.pth',
            'final_800_0.pth','final_900_0.pth','final_1000_0.pth'
            ]
    for i in pth_list:
        opt.pretrain = i
        adata, _ = test_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True, 
                    lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)
        print(adata.obsm['pred'].shape)

        sc.pp.neighbors(adata, use_rep='pred')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)


        sp2 = os.path.join(opt.result_path)
        if not os.path.exists(sp2):
            os.makedirs(sp2)

        sc.settings.figdir = sp2
        ax = sc.pl.umap(adata, color=['merge_cell_type'], show=False, title='combined latent variables')
        plt.savefig(os.path.join(sp2, 'umap_%s.pdf'%opt.pretrain[:-4]), bbox_inches='tight')

        adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()

        ## we use leiden algorithm for nanostring dataset
        # clustering the fovs
        if opt.cluster_method == 'leiden':
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            sc.pp.neighbors(adata, 8, use_rep='pred', random_state=seed)
                
            def res_search(adata_pred, ncluster, seed, iter=200):
                start = 0; end =3
                i = 0
                while(start < end):
                    if i >= iter: return res
                    i += 1
                    res = (start + end) / 2
                    print(res)
                    seed_everything(seed)
                    random.seed(seed)
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
                    count = len(set(adata_pred.obs['leiden']))
                    # print(count)
                    if count == ncluster:
                        print('find', res)
                        return res
                    if count > ncluster:
                        end = res
                    else:
                        start = res
                raise NotImplementedError()

            res = res_search(adata, 8, seed)
        
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # calculate ARI
            sc.tl.leiden(adata, resolution=res, key_added='leiden', random_state=seed)

            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['leiden'], obs_df['merge_cell_type'])
            print('ARI: %.2f'%ARI)

            cell_type = list(set(adata.obs['merge_cell_type']))
            ground_truth = [i for i in range(len(cell_type))]  
            gt = np.zeros(adata.obs['merge_cell_type'].shape)
            for i in range(len(ground_truth)):
                ct = cell_type[i]
                idx = (adata.obs['merge_cell_type'] == ct)
                gt[idx] = i
            gt = gt.astype(np.int32)

            pred = adata.obs['leiden'].to_numpy().astype(np.int32)
            layers = []
            cs = ['' for i in range(pred.shape[0])]
            gt_cs = ['' for i in range(pred.shape[0])]
            match = _hungarian_match(pred, gt, len(set(pred)), len(set(gt)))
            colors = {
                    'endothelial':"#26C5D9FF",
                    'epithelial':'#D3E057FF',
                    'fibroblast':'#26A599FF',
                    'lymphocyte':'#E57272FF',
                    'mast':'#5B6BBFFF', 
                    'myeloid':'#FFCA27FF',
                    'neutrophil':'#B967C7FF',
                    'tumors':'#A6CEE3',
                }
            cs = ['' for i in range(pred.shape[0])]
            gt_cs = ['' for i in range(pred.shape[0])]

            for ind, j in enumerate(adata.obs['merge_cell_type'].tolist()):
                gt_cs[ind] = colors[j]

            for outc, gtc in match:
                idx = (pred == outc)
                for j in range(len(idx)):
                    if idx[j]:
                        cs[j] = colors[cell_type[gtc]]
            adata.obs['cmap'] = cs
            adata.obs['gtcmap'] = gt_cs

            file_path = os.path.join(sp2,'process_data_%s.h5ad'%opt.pretrain[:-4])
            adata.write(file_path)

def train(opt, r=0):
    seed_everything(opt.seed)
    #load model
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'../model/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()

    ids = [
        'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
        'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
        'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
        'fov16', 'fov17', 'fov18', 'fov19', 'fov20'

    ]
    img_names = [
        'F001', 'F002', 'F003', 'F004', 'F005',
        'F006', 'F007', 'F008', 'F009', 'F010',
        'F011', 'F012', 'F013', 'F014', 'F015',
        'F016', 'F017', 'F018', 'F019', 'F020'
    ]

    adatas = list()
    for id, name in zip(ids, img_names):
        adata = gen_adatas(opt.root, id, name, model, opt.save_path)
        adatas.append(adata)

    sp = os.path.join(opt.save_path)
    if not os.path.exists(sp):
        os.makedirs(sp)

    file_path = os.path.join(sp, "adatas_list.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(adatas, file)

    train_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True,
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)


def train_nano(opt):
    if opt.test_only:
        infer(opt)
    else:
        train(opt, opt.repeat)
