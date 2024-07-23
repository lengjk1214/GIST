import argparse
import pandas as pd
import os
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils import Cal_Spatial_Net, Stats_Spatial_Net, _hungarian_match, seed_everything, mclust_R
from train_transformer import train_nano_fov, test_nano_fov
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import anndata
from scipy.sparse import csr_matrix
import pickle
from ctran import ctranspath



def gen_adatas(root, model=None):
    adata = sc.read(os.path.join(root, 'colorectal-cancer.h5ad'))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    del adata.obsm['imgs']

    img = cv2.imread(os.path.join(root, 'Colorectal_Cancer_image.tif'))
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.ToTensor()
    img = transform(img)

    patchs = []
    w, h = 120, 120

    drop = []
    for i, coor in enumerate(adata.obsm['spatial_scale']):
        x, y = coor

        if x < w or y < h:
            img_p = img[:, int(y-60):int(y+60), int(x-60): int(x+60)]
        else:
            img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]
        if not img_p.numel():
            if x > width:
                x = width - 10 
            if y > height:
                y = height - 10  
            img_p = img[:, int(y-10):int(y), int(x-10): int(x)]


        resize = transforms.Resize((224, 224))
        image_resized = resize(img_p)
        image_resized = image_resized.unsqueeze(0)
        with torch.no_grad():
            feature = model(image_resized)
            feature = feature.cpu().numpy()
        feature = torch.tensor(feature.squeeze())
        patchs.append(feature.squeeze())
        
    patchs = np.stack(patchs)
    
    adata.obsm['imgs'] = pd.DataFrame(patchs, index=adata.obs.index)


    Cal_Spatial_Net(adata, rad_cutoff=20, use_scale = True)
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

    # adatas = list()
    # adata = gen_adatas(opt.root, model)
    # adatas.append(adata)

    sp = os.path.join(opt.save_path)
    adatas = list()
    file_path = os.path.join(sp, "adata_list.pkl")
    with open(file_path, 'rb') as file:
        adata = pickle.load(file)
    adata = adata[0]
    adatas.append(adata)

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
        ax = sc.pl.umap(adata, color=['annotation'], show=False, title='combined latent variables')
        plt.savefig(os.path.join(sp2, 'umap_%s.pdf'%opt.pretrain[:-4]), bbox_inches='tight')

        adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()


        adata = mclust_R(adata, used_obsm='pred', num_cluster=opt.ncluster)
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['annotation'])
        print('ari is %.2f'%(ARI))
    

        # match the clusters to cell_types
        cell_type = list(set(adata.obs['annotation']))
        ground_truth = [i for i in range(len(cell_type))]
        gt = np.zeros(adata.obs['annotation'].shape)
        for i in range(len(ground_truth)):
            ct = cell_type[i]
            idx = (adata.obs['annotation'] == ct)
            gt[idx] = i
        gt = gt.astype(np.int32)

        pred = adata.obs['mclust'].to_numpy().astype(np.int32)
        layers = []
        cs = ['' for i in range(pred.shape[0])]
        gt_cs = ['' for i in range(pred.shape[0])]
        match = _hungarian_match(pred, gt, len(set(pred)), len(set(gt)))
        colors = {'desmoplastic changes': '#1f77b4',
                'muscularis propria': '#ff7f0e',
                'tumor': "#2ca02c",
                'tumor necrosis': "#d62728",
                'vessel':"#B967C7FF"
            }
        
        cs = ['' for i in range(pred.shape[0])]
        gt_cs = ['' for i in range(pred.shape[0])]

        for ind, j in enumerate(adata.obs['annotation'].tolist()):
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
    td = torch.load(r'../checkpoint/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()

    adatas = list()
    adata = gen_adatas(opt.root, model)
    adatas.append(adata)

    sp = os.path.join(opt.save_path)
    if not os.path.exists(sp):
        os.makedirs(sp)
    file_path = os.path.join(sp, "adata_list.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(adatas, file ,protocol = 4)

    train_nano_fov(opt, adatas, hidden_dims=[512, 30],  n_epochs=opt.epochs, save_loss=True,
                lr=opt.lr, random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=r)


def train_colorectal(opt):
    if opt.test_only:
        infer(opt)
    else:
        train(opt, opt.repeat)
