import gin
import os
import torch
import numpy as np
#import wandb

from accelerate import Accelerator
from data.processed_alt import ItemData, process_amazon_data
#from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import pandas as pd
from tqdm import tqdm
import json
import datetime
from pathlib import Path
import pickle
import torch._dynamo
torch._dynamo.config.suppress_errors = True


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    data_path='../data',
    #dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    save_dir_root="../outputs/rq_vae",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    log_metrics_every=1000,
    dataset_split="beauty",
    disable_compilation=False,  # Add this parameter
):
    #if wandb_logging:
       # params = locals()
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    save_dir_root = f"../outputs/rq_vae_{timestamp}"
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    
    device = accelerator.device
    
    #load data here 
    
    
    
    """

    train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all", split=dataset_split)
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval", split=dataset_split)
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all", split=dataset_split) if do_eval else train_dataset
    """
    data_path = Path(data_path)
    data = process_amazon_data(data_path) 
    train_dataset = ItemData(data, train_test_split="train")
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)
    eval_dataset = ItemData(data, train_test_split="eval")
    eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)
    index_dataset = ItemData(data, train_test_split="all")    
    
    train_dataloader = accelerator.prepare(train_dataloader)
    # TODO: Investigate bug with prepare eval_dataloader

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )

    # Set TorchInductor cache directory
    torchinductor_cache_dir = os.path.join(save_dir_root, "torchinductor_cache")
    os.makedirs(torchinductor_cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = torchinductor_cache_dir

    # Conditionally compile the model
    if not disable_compilation:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compilation enabled")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation")
    else:
        print("Model compilation disabled by configuration")

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

   # if wandb_logging and accelerator.is_main_process:
      #  wandb.login()
       # run = wandb.init(
       #     project="rq-vae-training",
       #     config=params
       # )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    
    tokenizer.rq_vae = model

    # Create metrics directory at the beginning
    metrics_dir = os.path.join(save_dir_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    train_metrics_file = os.path.join(metrics_dir, "train_metrics.json")
    val_metrics_file = os.path.join(metrics_dir, "val_metrics.json")

    # Initialize metrics storage
    all_train_metrics = []
    all_val_metrics = []

    with tqdm(initial=start_iter, total=start_iter+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], []]
        # Fix: Use range(start_iter, start_iter + iterations) for correct count
        for iter_idx in range(start_iter, start_iter + iterations):
            model.train()
            total_loss = 0
            t = 0.2
            if iter_idx == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(train_dataset[torch.arange(min(20000, len(train_dataset)))], device)
                model(kmeans_init_data, t)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter_idx % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process: #and wandb_logging:
                # Compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                }
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss.cpu().item(),
                    "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                    "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                    **emb_norms_avg_log,
                }

            if accelerator.is_main_process:
                # Collect training metrics every N iterations
                if (iter_idx + 1) % log_metrics_every == 0:
                    #print(f"Logging training metrics at iteration {iter_idx}")
                    train_metrics = {
                        "iteration": iter_idx,
                        "timestamp": datetime.datetime.now().isoformat(),
                        **train_log
                    }
                    all_train_metrics.append(train_metrics)

            if do_eval and ((iter_idx + 1) % eval_every == 0 or iter_idx + 1 == start_iter + iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter_idx+1}', disable=True) as pbar_eval:
                    eval_losses = [[], [], []]
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)

                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    
                if accelerator.is_main_process:
                    print(f"Logging validation metrics at iteration {iter_idx}")
                    val_metrics = {
                        "iteration": iter_idx,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "eval_total_loss": eval_losses[0],
                        "eval_reconstruction_loss": eval_losses[1],
                        "eval_rqvae_loss": eval_losses[2], 
                        "p_unique_ids": eval_model_output.p_unique_ids.cpu().item()
                    }
                    all_val_metrics.append(val_metrics)

            if accelerator.is_main_process:
                if (iter_idx+1) % save_model_every == 0 or iter_idx+1 == start_iter + iterations:
                    state = {
                        "iter": iter_idx,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter_idx}.pt")
                
                if (iter_idx+1) % eval_every == 0 or iter_idx+1 == start_iter + iterations:
                    tokenizer.reset()
                    model.eval()

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                    
                    _, counts = torch.unique(corpus_ids[:,:-1], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p*torch.log(p)).sum()

                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:,cid], return_counts=True)
                        id_diversity_log[f"codebook_usage_{cid}"] = len(counts) / vae_codebook_size

                    id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                    id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()
                
                # MOVE THE METRICS SAVING CODE HERE
                metrics_dir = os.path.join(save_dir_root, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                
                # Save training metrics
                #train_metrics_file = os.path.join(metrics_dir, f"train_metrics_{iter}.json")
               # with open(train_metrics_file, 'w') as f:
                 #   json.dump(train_log, f, indent=2)
               # print(f"Saved training metrics to {train_metrics_file}")
                
                # Save final diversity metrics
                #final_metrics_file = os.path.join(metrics_dir, f"final_metrics_{iter}.json")
               # with open(final_metrics_file, 'w') as f:
                #    json.dump(id_diversity_log, f, indent=2)
               # print(f"Saved diversity metrics to {final_metrics_file}")

            pbar.update(1)
    
    # After the training loop
    if accelerator.is_main_process:
        # Compute the diversity metrics at the end
        tokenizer.reset()
        model.eval()
        corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
        
        max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
        _, counts = torch.unique(corpus_ids[:,:-1], dim=0, return_counts=True)
        p = counts / corpus_ids.shape[0]
        rqvae_entropy = -(p*torch.log(p)).sum()
        
        id_diversity_log = {}
        for cid in range(vae_n_layers):
            _, counts = torch.unique(corpus_ids[:,cid], return_counts=True)
            id_diversity_log[f"codebook_usage_{cid}"] = len(counts) / vae_codebook_size
        
        id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
        id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()

        # Write the final_metrics.json
        final_metrics_file = os.path.join(metrics_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(id_diversity_log, f, indent=2)
        print(f"Saved final diversity metrics to {final_metrics_file}")
        corpus_ids = corpus_ids.detach().cpu().numpy()
        with open(data_path/'preprocessed/id_item_mapper.json', 'r') as stream:
            id_to_item = json.load(stream)
        final_sem_ids_file = data_path/"embeddings/my_beauty_sem_ids.npy"
        np.save(final_sem_ids_file, corpus_ids)
        sem_id_dict = {}
        for idx, item in id_to_item.items():
            sem_id_dict[item] = corpus_ids[int(idx)]
        with open(data_path/'embeddings/item_sem_id.pkl', 'wb') as stream:
            pickle.dump(sem_id_dict, stream)
        for item in sem_id_dict.keys():    
            for i in range(1, vae_n_layers + 1):
                sem_id_dict[item][i] += vae_codebook_size * i
        with open(data_path/'embeddings/item_sem_id_modified.pkl', 'wb') as stream:
            pickle.dump(sem_id_dict, stream)        
        print(f"Saved final sem_id-s metrics to {final_sem_ids_file}")
    #print(all_train_metrics)
    # Save all metrics at the end
    if accelerator.is_main_process:
        with open(train_metrics_file, 'w') as f:
            json.dump(all_train_metrics, f, indent=2)
        with open(val_metrics_file, 'w') as f:
            json.dump(all_val_metrics, f, indent=2)
        print(f"Saved training metrics to {train_metrics_file}")
        print(f"Saved validation metrics to {val_metrics_file}")

if __name__ == "__main__":
    parse_config()
    train()