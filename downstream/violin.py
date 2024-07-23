import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import pandas as pd
import numpy as np
from utils import _hungarian_match

path = "../results/lung13/process_data_final_200_0.h5ad"
path2 = "../results/lung13/200_violin_gt.pdf"
path3 = "../results/lung13/200_violin.pdf"
adata = sc.read(path)
print(adata)
adata = adata[adata.obs['merge_cell_type'].notna()]
pd.Series(adata.var_names).to_csv('var_index.csv', header=['var_index'], index=False)

marker_genes = [
    "VEGFA",'VEGFB', # Endothelial
    'EPCAM','KRT18', # Epithelial
    'VIM','PDGFRA', # Fibroblast
    'CD4','CD8A', # Lymphocyte
    'TPSAB1','KIT', # Mast
    'DNMT3A','CTSG', # Myeloid
    'CSF3R','CXCR2',# neutrophil
    'KRAS','ERBB2', # tumors
]

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
cs = ['' for i in range(pred.shape[0])]

for outc, gtc in match:
    idx = (pred == outc)
    for j in range(len(idx)):
        if idx[j]:
            cs[j] = cell_type[gtc]
adata.obs['leiden_color'] = cs


sc.pl.stacked_violin(adata, marker_genes, groupby='merge_cell_type', swap_axes=True, 
                    categories_order=['endothelial', 'epithelial', 'fibroblast', 'lymphocyte', 'mast', 'myeloid',
                         'neutrophil', 'tumors'],
                    figsize=[6,4],vmax = 1.5)
plt.savefig(path2, bbox_inches='tight')

recon_data = adata.copy()
recon_data.X = recon_data.layers['recon']
sc.pl.stacked_violin(recon_data, marker_genes, groupby='leiden_color', swap_axes=True, 
                    categories_order=['endothelial', 'epithelial', 'fibroblast', 'lymphocyte', 'mast', 'myeloid',
                         'neutrophil', 'tumors'],
                     figsize=[6,4],vmax = 1.5)
plt.savefig(path3, bbox_inches='tight')





