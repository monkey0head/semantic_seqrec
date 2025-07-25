import gin
import os
import random
import torch
import os
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from typing import NamedTuple
from sentence_transformers import SentenceTransformer

def create_embeds(meta_unique, data_path):
    meta_unique['full_description'] = 'Title: ' + meta_unique['title'] + '\n' + 'Description: ' + meta_unique['description']
    model_ckpt = 'intfloat/e5-base-v2'
    model = SentenceTransformer(model_ckpt)
    item_embeddings = model.encode(
        meta_unique['full_description'].fillna("").astype(str).to_list(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    os.makedirs(data_path/'embeddings', exist_ok = True)
    np.save(data_path/'embeddings/my_beauty_embs_all.npy', item_embeddings)
    return item_embeddings

def process_amazon_data(data_path):
    data = HeteroData()
    print("Loading shared data...")
    meta_df = pd.read_csv(data_path/'preprocessed'/'Beauty_desc_all.csv')
    if os.path.exists(data_path/'embeddings'/'my_beauty_embs_all.npy'):
        embeddings = np.load(data_path/'embeddings'/'my_beauty_embs_all.npy')
    else:
        embeddings = create_embeds(meta_df, data_path)
    item_ids = meta_df['item_id'].values
    item_emb = np.zeros((len(item_ids), embeddings.shape[1]))
    data['item'].x = torch.tensor(embeddings, dtype=torch.float)
    gen = torch.Generator()
    gen.manual_seed(42)
    data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
    return data


class ItemData(Dataset):
    def __init__(
        self,
        data,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
      
   
        if train_test_split == "train":
            filt = data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(data["item"]["x"][:,0], dtype=bool)
            
        self.item_data = data["item"]["x"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )