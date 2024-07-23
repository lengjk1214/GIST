import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from utils import _hungarian_match
from sklearn.metrics.cluster import adjusted_rand_score


path = '../results/lung13/process_data_final_0.h5ad'
path2 = "../results/lung13/umap.pdf"
path3 = "../results/lung13/raw_umap.pdf"

adata = sc.read(path)
adata = adata[adata.obs['merge_cell_type'].notna()]

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


del adata.uns['neighbors']
del adata.uns['umap']
sc.pp.neighbors(adata, use_rep='pred')
sc.tl.umap(adata)
plt.rcParams["figure.figsize"] = (3, 3)
ax = sc.pl.umap(adata, color=['merge_cell_type'],palette = colors,  show=False, title='combined latent variables')
plt.savefig(path2, bbox_inches='tight')

del adata.uns['neighbors']
del adata.uns['umap']
sc.pp.neighbors(adata,use_rep='X')
sc.tl.umap(adata)
plt.rcParams["figure.figsize"] = (3, 3)
ax = sc.pl.umap(adata, color=['merge_cell_type'],palette = colors,  show=False, title='combined latent variables')
plt.savefig(path3, bbox_inches='tight')
