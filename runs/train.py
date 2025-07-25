"""
Run experiment.
"""

import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)

from transformers import GPT2Config, GPT2LMHeadModel

from torch import nn
from torch.utils.data import DataLoader

from src.datasets import (CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn)
from src.metrics import Evaluator
from src.models import SASRec
from src.modules import SeqRec, SeqRecHuggingface
from src.postprocess import preds2recs
from src.prepr import last_item_split

import pickle
from tqdm.auto import tqdm


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.clearml_project_folder is not None:
        split_subtype = config.split_subtype or ''
        project_name = os.path.join(
            config.clearml_project_folder, config.split_type, split_subtype,
            config.dataset.name, config.model.model_class)
        task = Task.init(project_name=project_name, task_name=config.clearml_task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config, resolve=True))
    else:
        task = None

    fix_seeds(config.random_state)
    
    train, validation, test, max_item_id, global_timepoint, global_timepoint_val = prepare_data(config)
    
    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=max_item_id)
    trainer, seqrec_module, num_best_epoches = training(model, train_loader, eval_loader, config, task)
        
    if trainer is not None:
        recs_validation = run_eval(validation, trainer, seqrec_module, config, max_item_id, task, global_timepoint_val, prefix='val')
    else:
        print('Skipping validation') 

    test_prefix = 'test'
    if config.retrain_with_validation:
        merged = pd.concat([train, validation], ignore_index=True)
        merged_loader, _ = create_dataloaders(merged, pd.DataFrame([], columns=validation.columns), config)
        model = create_model(config, item_count=max_item_id)
        config.trainer_params.max_epochs = num_best_epoches
        trainer, seqrec_module, _ = training(model, merged_loader, None, config, task, retrain=True)
        test_prefix = 'test_retrained'
    
    recs_test = run_eval(test, trainer, seqrec_module, config, max_item_id, task, global_timepoint, prefix=test_prefix)

    # recs_test.to_csv(f'data/splitted/{config.dataset.name}.csv')

    if task is not None:
        task.close()

def fix_seeds(random_state):
    """Set up random seeds."""

    seed_everything(random_state, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(config):
    split_subtype = config.split_subtype or ''
    split_type = config.split_type
    if split_type == 'global_timesplit':
        if config.quantile is not None:
            q = 'q0'+str(config.quantile)[2:]
        else:
            raise ValueError("'global_timesplit' split must be run with parameter 'quantile'")
    else:
        q = ''
    data_path = os.path.join(
        os.environ["SEQ_SPLITS_DATA_PATH"], 'splitted', split_type,
        split_subtype, config.dataset.name, q)
    

    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    print('train shape', train.shape)
    validation = pd.read_csv(os.path.join(data_path, 'validation.csv'))
    print('validation shape', validation.shape)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    print('test shape', test.shape)

    if config.use_semantic_ids and config.semantic_ids_map_path is not None:
        with open(config.semantic_ids_map_path, 'rb') as f:
            index2semid = pickle.load(f)

        assert len(list(index2semid.values())[0]) == config.semantic_ids_len, f"Semantic IDs length mismatch: {len(list(index2semid.values())[0])} != {config.semantic_ids_len}"

        train.item_id = train.item_id.map(index2semid)
        validation.item_id = validation.item_id.map(index2semid)
        test.item_id = test.item_id.map(index2semid)

        print(train.head(5))
        print(test.head(5))
        print(validation.head(5))

        train = train.explode('item_id')
        validation = validation.explode('item_id')
        test = test.explode('item_id')

        print(train.head(5))
        print(test.head(5))
        print(validation.head(5))

    max_item_id = max(train.item_id.max(), test.item_id.max(), validation.item_id.max())

    global_timepoint = None
    global_timepoint_val = None

    if split_type == 'global_timesplit':
        with open(os.path.join(data_path,'time_threshold.pkl'), 'rb') as f:
            global_timepoint = pickle.load(f)
        print('Test global timepoint', global_timepoint)

        if split_subtype == 'val_by_time':
            with open(os.path.join(data_path,'val_time_threshold.pkl'), 'rb') as f:
                global_timepoint_val = pickle.load(f)
            print('Validation global timepoint', global_timepoint_val)

    return train, validation, test, max_item_id, global_timepoint, global_timepoint_val


def create_dataloaders(train, validation, config):

    if config.model.model_class=='GPT-2':
        shift_labels = False
    else:
        shift_labels = True

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()

    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]

    semantic_ids_len = config.semantic_ids_len if config.use_semantic_ids else 1
    max_length = config.dataset_params.max_length * semantic_ids_len
    
    train_dataset = CausalLMDataset(train, max_length=max_length, semantic_ids_len=semantic_ids_len, shift_labels=shift_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())


    if config.model.generation:
        # eval_dataset = CausalLMPredictionDataset(validation, max_length=max_length, semantic_ids_len=semantic_ids_len, validation_mode=True)

        eval_dataset = CausalLMPredictionDataset(
            validation, max_length=(config.dataset_params.max_length - max(config.evaluator.top_k)) * semantic_ids_len,
            semantic_ids_len=semantic_ids_len, validation_mode=True)
        
        eval_loader = DataLoader(
                eval_dataset, shuffle=False,
                collate_fn=PaddingCollateFn(left_padding=True),
                batch_size=config.dataloader.test_batch_size,
                num_workers=config.dataloader.num_workers)

    else:
        eval_dataset = CausalLMPredictionDataset(validation, max_length=max_length, semantic_ids_len=semantic_ids_len, validation_mode=True)
        eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                                shuffle=False, num_workers=config.dataloader.num_workers,
                                collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def add_padding_embedding(item_embeddings, pad_token_id=0):
    return torch.vstack(
        (
            item_embeddings[: pad_token_id],
            torch.zeros(item_embeddings[0].shape),
            item_embeddings[pad_token_id :],
        )
    )

def create_model(config, item_count):
    if config.model.model_class == 'GPT-2':
        gpt2_config = GPT2Config(vocab_size=item_count + 1, **config.model.model_params)

        if config.use_pretrained_embeddings and config.dataset.pretrained_embeddings_path is not None:
            embeddings_tensor = torch.FloatTensor(
                np.load(config.dataset.pretrained_embeddings_path)
            )

            if config.pretrained_embeddings.add_padding_emb:
                embeddings_tensor = add_padding_embedding(embeddings_tensor, 0)

            gpt2_config.n_embd = embeddings_tensor.shape[-1] #to properly initialize positional embeddings
            model = GPT2LMHeadModel(gpt2_config)

            model.wte = nn.Embedding.from_pretrained(
                embeddings_tensor,
                freeze = config.pretrained_embeddings.freeze, 
                padding_idx = 0
            )

            print('\n-Pretrained embeddings are used:', model.wte)
            print('---Freeze:', config.pretrained_embeddings.freeze)

        else:
            model = GPT2LMHeadModel(gpt2_config)
            
    elif config.model.model_class == 'SASRec':
        if config.use_pretrained_embeddings and config.dataset.pretrained_embeddings_path is not None:
            
            embeddings_tensor = torch.FloatTensor(
                np.load(config.dataset.pretrained_embeddings_path)
            )

            if config.pretrained_embeddings.add_padding_emb:
                embeddings_tensor = add_padding_embedding(embeddings_tensor, 0)

            config.model.model_params.hidden_units = embeddings_tensor.shape[-1] #to properly initialize positional embeddings
            model = SASRec(item_num=item_count, **config.model.model_params)

            model.item_emb = nn.Embedding.from_pretrained(
                embeddings_tensor,
                freeze = config.pretrained_embeddings.freeze, 
                padding_idx = model.padding_idx
            )

            print('\n-Pretrained embeddings are used:', model.item_emb)
            print('--Freeze:', config.pretrained_embeddings.freeze)
        else:
            model = SASRec(item_num=item_count, **config.model.model_params)

    return model


def training(model, train_loader, eval_loader, config, task=None, retrain=False):

    split_subtype = config.split_subtype or ''
    q = 'q0' + str(config.quantile)[2:] if config.split_type == 'global_timesplit' else ''
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', config.split_type,
        split_subtype, config.dataset.name, q, config.model.model_class, 'retrain_with_val' if retrain else '')
    
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if config.model.model_class == 'SASRec':
        file_name = (
            f"{config.model.model_params.hidden_units}_"
            f"{config.model.model_params.num_blocks}_"
            f"{config.model.model_params.num_heads}_"
            f"{config.model.model_params.dropout_rate}_"
            f"{config.model.model_params.maxlen}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}"
        )
    elif config.model.model_class == 'GPT-2':
        file_name = (
            f"{config.model.model_params.n_embd}_"
            f"{config.model.model_params.n_layer}_"
            f"{config.model.model_params.n_head}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}"
        )
    checkpoint_file = os.path.join(model_path, file_name + ".ckpt")

    if config.model.model_class == 'GPT-2':
        seqrec_module = SeqRecHuggingface(model, **config['seqrec_module'])
        if config.model.generation:
            with open(config.semantic_ids_map_path, 'rb') as f:
                index2semid = pickle.load(f)
            inv_map = {tuple(sem_ids): item_id for item_id, sem_ids in index2semid.items()}
            seqrec_module.set_predict_mode(generate=True, mode=config.model.mode,
                                           N=config.semantic_ids_len,
                                           inv_map=inv_map,
                                           **config.model.generation_params)
    else:   
        seqrec_module = SeqRec(model, **config['seqrec_module']) 
    
    if getattr(config, 'load_if_possible', False) and os.path.exists(checkpoint_file):
        
        checkpoint_dict = torch.load(checkpoint_file)
        seqrec_module.load_state_dict(checkpoint_dict['state_dict'])
        num_best_epoches = checkpoint_dict['epoch'] + 1
        
        print(f'Loaded trained model from: {checkpoint_file}')
        return None, seqrec_module, num_best_epoches
    
    model_summary = ModelSummary(max_depth=1)
    progress_bar = TQDMProgressBar(refresh_rate=20)

    if not retrain:
        checkpoint = ModelCheckpoint(
            dirpath=model_path,  
            filename='_' + file_name,           
            save_top_k=1,
            monitor="val_ndcg",
            mode="max",
            save_weights_only=True
        )
        early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                    patience=config.patience, verbose=False)
        callbacks = [early_stopping, model_summary, checkpoint, progress_bar]
    
    else:
        checkpoint = ModelCheckpoint(
            dirpath=model_path,  
            filename='_' + file_name,           
            save_weights_only=True
        )
        callbacks = [model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True, 
                         **config['trainer_params'])
    
    start_time = time.time()

    try:
        trainer.fit(model=seqrec_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=eval_loader)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if os.path.exists(checkpoint.best_model_path):
                os.remove(checkpoint.best_model_path)
                print(f"Removed checkpoint due to CUDA OOM error: {checkpoint.best_model_path}")
        raise
    
    finally:
        if not getattr(trainer, "interrupted", False):
            seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])
            os.rename(checkpoint.best_model_path, checkpoint_file)
            print(f"Checkpoint renamed to: {checkpoint_file}")
            num_best_epoches = torch.load(checkpoint_file)['epoch'] + 1
        else:
            print("Detected interruption of training.")
            os.remove(checkpoint.best_model_path)

    training_time = time.time() - start_time
    print('training_time', training_time)

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)

    return trainer, seqrec_module, num_best_epoches


def predict(trainer, seqrec_module, data, config, global_timepoint):

    semantic_ids_len = config.semantic_ids_len if config.use_semantic_ids else 1


    if config.model.model_class == 'GPT-2':
        if config.model.generation:
            predict_dataset = CausalLMPredictionDataset(
                data, max_length=(config.dataset_params.max_length - max(config.evaluator.top_k)) * semantic_ids_len,
                semantic_ids_len=semantic_ids_len)
            
            predict_loader = DataLoader(
                    predict_dataset, shuffle=False,
                    collate_fn=PaddingCollateFn(left_padding=True),
                    batch_size=config.dataloader.test_batch_size,
                    num_workers=config.dataloader.num_workers)
            
            inv_map = None
            if config.use_semantic_ids and config.semantic_ids_map_path is not None:
                with open(config.semantic_ids_map_path, 'rb') as f:
                    index2semid = pickle.load(f)
                inv_map = {tuple(sem_ids): item_id for item_id, sem_ids in index2semid.items()}

            seqrec_module.set_predict_mode(generate=True, mode=config.model.mode,
                                           N=semantic_ids_len,
                                           inv_map=inv_map,
                                           **config.model.generation_params)

        else:
            predict_dataset = CausalLMPredictionDataset(data, 
                                            max_length=config.dataset_params.max_length  * semantic_ids_len,
                                            semantic_ids_len=semantic_ids_len)

            predict_loader = DataLoader(
                    predict_dataset, shuffle=False,
                    collate_fn=PaddingCollateFn(),
                    batch_size=config.dataloader.test_batch_size,
                    num_workers=config.dataloader.num_workers)
            
            seqrec_module.set_predict_mode(generate=False)

    elif config.model.model_class == 'SASRec':
        if config.use_semantic_ids:
            raise NotImplementedError("Semantic IDs are not supported for SASRec")
        predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset_params.max_length)

        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=config.dataloader.test_batch_size, 
            num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)

    if trainer is None:
        predict_trainer_params = config.get('trainer_predict_params', {})
        trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=100)], **predict_trainer_params)

    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
    recs = preds2recs(preds, successive=False)
    print('recs shape', recs.shape)

    return recs


def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def unexplode(df, N=4, auxiliary_column='timestamp'):
    grouped = df.groupby('user_id').agg({
        'item_id': list,
        auxiliary_column: list,
    })
    
    grouped['item_id'] = grouped['item_id'].apply(chunker, size=N)
    grouped[auxiliary_column] = grouped[auxiliary_column].apply(chunker, size=N)
    
    result = grouped.explode(['item_id', auxiliary_column])
    result[auxiliary_column] = result[auxiliary_column].apply(lambda x: x[0])

    result['item_id'] = result['item_id'].apply(lambda x: tuple(x))
    
    return result.reset_index()


def run_eval(data, trainer, seqrec_module, config, max_item_id, task, global_timepoint, prefix):
    """Get predictions and ground truth for selected ground truth type. Calculate metrics."""
    start_time = time.time()
    
    test_ground_truth = None
    recs = None

    if config.use_semantic_ids:
        data = unexplode(data)

    test, test_ground_truth = last_item_split(data)

    if config.use_semantic_ids:
        test = test.explode('item_id')

    recs = predict(trainer, seqrec_module, test, config, global_timepoint)

    # if config.use_semantic_ids:
    #     recs = unexplode(recs, auxiliary_column='prediction')

    if config.split_type != 'leave-one-out':
        prefix = f'{prefix}_last'
    
    evaluate(recs, test_ground_truth, task, config, max_item_id, prefix=prefix)

    eval_time = time.time() - start_time
    print(f"{prefix} predict and evaluation time", eval_time)

    if task is not None:
        task.get_logger().report_single_value(f"{prefix}_time", eval_time)
        if config[f"save_{prefix}_predictions"]:
            task.upload_artifact(f"{prefix}_pred.csv", recs)

    return recs

def evaluate(recs, ground_truth, task, config, num_items=None, prefix='test'):    
    all_items = None
    if "Coverage" in config.evaluator.metrics:
        all_items = pd.DataFrame({"item_id": np.arange(1, num_items + 1)})    
        # to pass replay data check
        all_items["user_id"] = 0

    evaluator = Evaluator(metrics=list(config.evaluator.metrics),
                        top_k=list(config.evaluator.top_k))
    metrics = evaluator.compute_metrics(ground_truth, recs, train=all_items)
    print(f'{prefix} metrics:\n', metrics_dict_to_df(metrics, config), '\n')
    metrics = {prefix + '_' + key: value for key, value in metrics.items()}

    if task:
        clearml_logger = task.get_logger()
        for key, value in metrics.items():
            clearml_logger.report_single_value(key, value)
        metrics = pd.Series(metrics).to_frame().reset_index()
        metrics.columns = ['metric_name', 'metric_value']

        clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                    table_plot=metrics)
        task.upload_artifact(f'{prefix}_metrics', metrics)
    else:
        metrics = pd.Series(metrics).to_frame().reset_index()
        metrics.columns = ['metric_name', 'metric_value']
        save_local_metrics(metrics, config, prefix)

def metrics_dict_to_df(d, config):

    metrics = config.evaluator.metrics
    ks = config.evaluator.top_k
    data = {
        metric: [
            d.get(f"{'HitRate' if metric=='HR' else metric}@{k}", float('nan'))
            for k in ks
        ]
        for metric in metrics
    }
    df = pd.DataFrame(data, index=[f"@{k}" for k in ks]).T.reindex(metrics)
    
    return df.rename(index={"HitRate": "HR", "Coverage": "COV"})

def save_local_metrics(metrics, config, prefix):
    
    split_subtype = config.split_subtype or ''
    q = 'q0'+str(config.quantile)[2:] if config.split_type == 'global_timesplit' else ''
    results_path = os.path.join(
        os.environ["SEQ_SPLITS_DATA_PATH"], 'results', config.split_type,
        split_subtype, config.dataset.name, q, config.model.model_class, prefix)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if config.model.model_class == 'SASRec':
        file_name = (
            f"{config.model.model_params.hidden_units}_"
            f"{config.model.model_params.num_blocks}_"
            f"{config.model.model_params.num_heads}_"
            f"{config.model.model_params.dropout_rate}_"
            f"{config.model.model_params.maxlen}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}.csv"
        )
    elif config.model.model_class == 'GPT-2':
        file_name = (
            f"{config.model.model_params.n_embd}_"
            f"{config.model.model_params.n_layer}_"
            f"{config.model.model_params.n_head}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}.csv"
        )

    metrics.to_csv(results_path + '/' + file_name, index=True)


if __name__ == "__main__":

    main()
