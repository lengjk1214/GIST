import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils import _hungarian_match
import math


path = '../results/lung13/process_data_final_0.h5ad'
path2 = "../results/lung13/gt.png"
path3 = "../results/lung13/model.png"
adata = sc.read(path)
cxg, cyg = adata.obs['cx'], adata.obs['cy']
print(adata)
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



fov_list =[3332, 3971, 3430, 2963, 4339,
    2999, 3719, 3666, 3822, 4029,
    4244, 3467, 3918, 4480, 4272,
    3654, 4191, 4659, 4192, 3696]
new_order = [16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3]
colors = [ '#26C5D9FF', '#D3E057FF', '#26A599FF', '#E57272FF', '#5B6BBFFF', '#FFCA27FF', 
          '#B967C7FF', '#A6CEE3'] 
color_labels = ['endothelial', 'epithelial', 'fibroblast', 'lymphocyte', 'mast', 'myeloid',
                         'neutrophil', 'tumors']
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, color_labels)]


begin = 0
end = 0
fig,axes=plt.subplots(6,5, figsize=(12, 10))
                   
plt.subplots_adjust(left=0.05,top=0.95,wspace=0,hspace=0)
color = adata.obs['gtcmap']
for idx, value in enumerate(new_order):
    ax = axes.flat[value]
    cur_idx = fov_list[idx]
    end = end + cur_idx
    subset_cxg = cxg[begin:end] 
    subset_cyg = cyg[begin:end]
    colors = color[begin:end]
    ax.invert_yaxis()
    ax.scatter(subset_cxg, subset_cyg, c=colors, s=1)  
    ax.axis('off')
    
    begin = begin + cur_idx
# plt.legend(bbox_to_anchor=(1.04,1), loc="center left",handles=legend_patches)
plt.savefig(path2, bbox_inches='tight')


begin = 0
end = 0
fig,axes=plt.subplots(6,5, figsize=(12, 10))
plt.subplots_adjust(left=0.05,top=0.95,wspace=0,hspace=0)
color = adata.obs['cmap']
for idx, value in enumerate(new_order):
    ax = axes.flat[value]
    cur_idx = fov_list[idx]
    end = end + cur_idx
    subset_cxg = cxg[begin:end] 
    subset_cyg = cyg[begin:end]
    colors = color[begin:end]
    ax.invert_yaxis()
    ax.scatter(subset_cxg, subset_cyg, c=colors, s=1)  
    ax.axis('off')
    
    begin = begin + cur_idx

# plt.legend(bbox_to_anchor=(1.04,1), loc="center left",handles=legend_patches) 
plt.savefig(path3, bbox_inches='tight')






