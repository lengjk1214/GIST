import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.loader import DataLoader

from transModel import TransImg
from sklearn.metrics.cluster import adjusted_rand_score


from utils import Transfer_img_Data, seed_everything, mclust_R
from sklearn.decomposition import PCA

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scanpy as sc
import os



@torch.no_grad()
def test_nano_fov(opt, adatas, model_name=None, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, 
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1):
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    datas, imgs = [], []
    gt_frame = None
    datas, gene_dims = [], []
    gene_dim = 0
    img_dim = 0 # gene and img dim is same for all fovs

    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        # print(data.x.shape, img.x.shape)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        datas.append(data)
    import anndata
    adata = anndata.concat(adatas)

    loader = DataLoader(datas, batch_size=1, shuffle=False)
    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)
    # torch.save(model.state_dict(), os.path.join(save_path, 'init.pth'))
    if model_name is not None:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
    else:
        print(os.path.join(save_path, opt.pretrain))
        model.load_state_dict(torch.load(os.path.join(save_path, opt.pretrain)))

    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hidden_matrix = None
    gene_matrix = None
    img_matrix = None
    couts = None
    losses = 0
    for batch in loader:
        batch = batch.to(device)
        # print(batch)
        bgene = batch.x[:, :gene_dim]
        bimg = batch.x[:, gene_dim:]
        # exit(0)
        edge_index = batch.edge_index
        gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)
        gloss = F.mse_loss(bgene, gout)
        iloss = F.mse_loss(bgene, iout)
        closs = F.mse_loss(bgene, cout)
        loss = (gloss + iloss + closs)
        losses += loss.item()

        print(cz.shape)
        if hidden_matrix is None:
            hidden_matrix = cz.detach().cpu()
            gene_matrix = gz.detach().cpu()
            couts = cout.detach().cpu()
            img_matrix = iz.detach().cpu()
        else:
            hidden_matrix = torch.cat([hidden_matrix, cz.detach().cpu()], dim=0)
            gene_matrix = torch.cat([gene_matrix, gz.detach().cpu()], dim=0)
            img_matrix = torch.cat([img_matrix, iz.detach().cpu()], dim=0)
            couts = torch.cat([couts, cout.detach().cpu()], dim=0)
    # exit(0)
    hidden_matrix = hidden_matrix.numpy()
    gene_matrix = gene_matrix.numpy()
    img_matrix = img_matrix.numpy()
    adata.obsm['pred'] = hidden_matrix
    adata.obsm['gene_pred'] = gene_matrix
    adata.obsm['img_pred'] = img_matrix
    couts = couts.numpy().astype(np.float32)
    couts[couts < 0] = 0
    adata.layers['recon'] = couts
    print(losses)
    return adata, losses


def train_nano_fov(opt, adatas, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, 
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_path='../checkpoint/trans_gene/', ncluster=7, repeat=1,
                gene_weight=0.1, img_weight=0.1, combine_weight=1.0):
    # seed_everything(random_seed)
    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    datas, gene_dims = [], []
    gene_dim = 0
    img_dim = 0 # gene and img dim is same for all fovs
    for adata in adatas:
        adata.X = sp.csr_matrix(adata.X)
        data, img = Transfer_img_Data(adata)
        gene_dim = data.x.shape[1]
        img_dim = img.x.shape[1]
        data.x = torch.cat([data.x, img.x], dim=1)
        datas.append(data)
    loader = DataLoader(datas, batch_size=1, shuffle=True)

    model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    seed = random_seed
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    for epoch in tqdm(range(1, n_epochs+1)):
        for i, batch in enumerate(loader):
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad()
            bgene = batch.x[:, :gene_dim]
            bimg = batch.x[:, gene_dim:]
            edge_index = batch.edge_index
            gz,iz,cz, gout,iout,cout = model(bgene, bimg, edge_index)

            gloss = F.mse_loss(bgene, gout)
            iloss = F.mse_loss(bgene, iout)
            closs = F.mse_loss(bgene, cout)
            loss = gene_weight * gloss + img_weight * iloss + combine_weight * closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        
        if epoch > 100 and epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'final_%d_%d.pth'%(epoch, repeat)))


    torch.save(model.state_dict(), os.path.join(save_path, 'final_%d.pth'%(repeat)))
    return adata

