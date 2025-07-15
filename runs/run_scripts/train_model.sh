python runs/train_necessary_only.py \
  model=GPT2 \
  dataset=Zvuk \
  split_type=global_timesplit \
  split_subtype=val_by_time \
  quantile=0.9 \
  cuda_visible_devices=0 \
  seqrec_module.predict_top_k=10 \
  use_semantic_ids=true\