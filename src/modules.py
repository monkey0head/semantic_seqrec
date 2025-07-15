"""
Pytorch Lightning Modules.
"""

from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from transformers.generation.logits_process import LogitsProcessor

class SeqRecBase(pl.LightningModule):

    def __init__(self, model, lr=1e-3, padding_idx=0,
                 predict_top_k=10, filter_seen=True):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        
            preds, scores = self.make_prediction(batch)

            scores = scores.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            user_ids = batch['user_id'].detach().cpu().numpy()
            
            if 'target_ids' in batch:
                target_ids = batch['target_ids'].detach().cpu().numpy()
                return {'preds': preds,
                        'scores': scores,
                        'user_ids': user_ids,
                        'target_ids': target_ids}
            
            return {'preds': preds,
                    'scores': scores,
                    'user_ids': user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)
    
    def make_prediction(self, batch):
        
        outputs = self.prediction_output(batch)

        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1
        preds = outputs[rows_ids, last_item_idx, :]
        
        if self.filter_seen:
            seen_items = batch['seen_ids'] 
            seen_mask = torch.zeros_like(preds, dtype=torch.bool)
            seen_mask.scatter_(1, seen_items.clamp(min=0, max=preds.shape[1]), True)  
            preds = preds.masked_fill(seen_mask, float("-inf"))

        scores, preds = torch.topk(preds, self.predict_top_k, dim=1) 
        return preds, scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}


class SeqRec(SeqRecBase):

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, outputs, batch):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

        return loss

    def prediction_output(self, batch):

        return self.model(batch['input_ids'], batch['attention_mask'])


class SeqRecHuggingface(SeqRecBase):
    generate: bool = False
    generate_params: dict

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def prediction_output(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        return outputs.logits

    def set_predict_mode(self,
                         generate=False,
                         mode='reciprocal_rank_aggregation',
                         **generate_kwargs):
        """
        Set `predict` options.
        If `generate` is False, the general predict method is used, which returns top-k most relevant next items.
        If `generate` is True, the sequence is continued with the `generate` method of the HuggingFaceModel class.
        `generate_kwargs` are passed to model.generate().
        `Generate` params are explained here: https://huggingface.co/blog/how-to-generate.
        If 'mode' is 'reciprocal_rank_aggregation', for each item sum reciprocal ranks across all sequences; also used for greedy search, beam search and temperature sampling with a single sequence, because it takes into account only the order of items in the generated sequence.
        If 'mode' is 'relevance_aggregation', for each item sum relevances across all steps, then across all sequences.
        """
        self.mode = mode
        self.generate = generate
        self.generate_params = {"early_stopping": False,
                                'top_k': 0,
                                'return_dict_in_generate': True,
                                'output_scores': True
                                } if (self.mode != 'reciprocal_rank_aggregation' and self.generate) else {"early_stopping": False}

        if generate and generate_kwargs is not None:
            self.generate_params.update(generate_kwargs)

    def process_multiple_sequences(self, batch, preds, scores):
        """
        Combine multiple sequences generated for one user into one and leave top-k with maximal score.
        Score of an item is calculated as a sum of scores of an item in each sequence.
        """
        num_seqs = self.generate_params["num_return_sequences"]

        if self.mode == 'relevance_aggregation':
            summed_scores = scores.reshape(batch['user_id'].shape[0], num_seqs, -1).sum(dim=1).sort(descending=True)
            scores_batch = summed_scores[0][:, :self.predict_top_k].cpu()
            preds_batch = summed_scores[1][:, :self.predict_top_k].cpu()
        else:
            preds_batch, scores_batch = [], []
            for user_idx in range(batch['user_id'].shape[0]):
                dicts = [dict(zip(preds[user_idx * num_seqs + i, :].detach().cpu().numpy(),
                                  scores[user_idx * num_seqs + i, :].detach().cpu().numpy())) for i in range(num_seqs)]
                combined_dict = dict(sum((Counter(d) for d in dicts), Counter()))
                preds_one, scores_one = list(
                    zip(*sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)[:preds.shape[1]]))
                preds_batch.append(preds_one)
                scores_batch.append(scores_one)

        return np.array(preds_batch), np.array(scores_batch)

    def predict_step(self, batch, batch_idx):
        user_ids = batch['user_id'].detach().cpu().numpy()

        if not self.generate or \
                ("num_return_sequences" not in self.generate_params
                 or ((self.generate_params["num_return_sequences"] == 1) & (self.mode != 'relevance_aggregation'))):
            if not self.generate:
                preds, scores = self.make_prediction(batch)
            else:
                preds, scores = self.make_prediction_generate(batch)
            scores = scores.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

        # `generate` with num_return_sequences > 1
        preds, scores = self.make_prediction_generate(batch)
        preds, scores = self.process_multiple_sequences(batch, preds, scores)
        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}
    
    def validation_step(self, batch, batch_idx):

        if not self.generate:
            preds, scores = self.make_prediction(batch)
        else:
            preds, scores = self.make_prediction_generate(batch)

        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)

    def make_prediction_generate(self, batch):
        """
        Continue the sequence with the `generate` method of the HuggingFaceModel class.
        Batch should be left-padded, e.g., with the PaddingCollateFn(left_padding=True).
        Input sequence may be cropped,
        maximum self.model.config.n_positions - self.predict_top_k last items are used as a sequence beginning.
        """
        if self.mode == 'reciprocal_rank_aggregation':
            seq = self.model.generate(
                batch['input_ids'][:, - self.model.config.n_positions + self.predict_top_k:].to(self.model.device),
                pad_token_id=self.padding_idx,
                max_new_tokens=self.predict_top_k,
                logits_processor=[FilterSeenProcessor()],
                **self.generate_params
            )
            preds = seq[:, -self.predict_top_k:]

            scores_one = torch.pow(
                torch.arange(
                    self.predict_top_k,
                    dtype=torch.long,
                    device=self.model.device
                ) + 1., -1
            ).reshape(1, -1)
            scores = torch.tile(scores_one, [preds.shape[0], 1])

        else:
            seq = self.model.generate(
                batch['input_ids'][:, -self.model.config.n_positions + self.predict_top_k:].to(self.model.device),
                pad_token_id=self.padding_idx,
                max_new_tokens=self.predict_top_k,
                **self.generate_params
            )

            preds = None

            if 'temperature' in self.generate_params.keys():
                temp = self.generate_params['temperature']
                scores = torch.nn.functional.softmax(torch.stack(list(seq.scores), dim=0) / temp, dim=-1).sum(dim=0)
            else:
                scores = torch.nn.functional.softmax(torch.stack(list(seq.scores), dim=0), dim=-1).sum(dim=0)

            scores = scores.scatter_(1, batch['full_history'].repeat_interleave(self.generate_params["num_return_sequences"], dim=0).to(self.model.device), -torch.inf)
        
        # print(preds)

        return preds, scores

class FilterSeenProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        return scores.scatter_(1, input_ids, -float('inf'))