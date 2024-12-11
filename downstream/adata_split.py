import scanpy as sc
import pandas as pd


path = '../results/lung13/process_data_final_0.h5ad'
adata = sc.read(path)
adata = adata[adata.obs['merge_cell_type'].notna()]

index_values = adata.obs.index
adata_list = []


for start_number in pd.Index(index_values).str.split('_').str[0].unique():
    subset_indices = index_values[pd.Index(index_values).str.startswith(f"{start_number}_")]
    adata_subset = adata[subset_indices.tolist()].copy()
    adata_list.append(adata_subset)
    print(len(subset_indices))

print(f"adata_list-{len(adata_list)}")
