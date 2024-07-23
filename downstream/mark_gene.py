import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os


 
path = '../results/lung13/process_data_final_0.h5ad'
path2 = "../results/lung13/mk_gt_KRAS"
path3 = "../results/lung13/mk_CLIFE_KRAS"

adata = sc.read(path)
print(adata)
adata = adata[adata.obs['merge_cell_type'].notna()]

fov_list =[3332, 3971, 3430, 2963, 4339,
    2999, 3719, 3666, 3822, 4029,
    4244, 3467, 3918, 4480, 4272,
    3654, 4191, 4659, 4192, 3696]
new_order = [16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3]
coor = adata.obsm['spatial']
cxg, cyg = coor[:,0], coor[:,1]

gene_marker = 'KRAS'
recon_data = adata.copy()
recon_data.X = recon_data.layers['recon']

rawv = adata[:, gene_marker].X.A
reconv = recon_data[:, gene_marker].X


begin = 0
end = 0
figsize = (4, 4)
color = rawv
for idx, value in enumerate(new_order):
    fig, ax = plt.subplots(figsize=figsize)
    cur_idx = fov_list[idx]
    end = end + cur_idx
    subset_cxg = cxg[begin:end] 
    subset_cyg = cyg[begin:end]
    colors = color[begin:end]
    ax.invert_yaxis()
    scatter1 = ax.scatter(subset_cxg, subset_cyg, c=colors, vmax=2, s=2, cmap = 'viridis')  
    cbar1 = plt.colorbar(scatter1, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    begin = begin + cur_idx
    save_path = os.path.join(path2, f"gt_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close()  

begin = 0
end = 0
color = reconv
for idx, value in enumerate(new_order):
    fig, ax = plt.subplots(figsize=figsize)
    cur_idx = fov_list[idx]
    end = end + cur_idx
    subset_cxg = cxg[begin:end] 
    subset_cyg = cyg[begin:end]
    colors = color[begin:end]
    ax.invert_yaxis()
    scatter2 = ax.scatter(subset_cxg, subset_cyg, c=colors, vmax=2,s=2, cmap = 'viridis') 
    cbar2 = plt.colorbar(scatter2, ax=ax, fraction=0.046, pad=0.04) 
    ax.axis('off')
    begin = begin + cur_idx
    save_path = os.path.join(path3, f"ctrans_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close()  







