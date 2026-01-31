CUDA_VISIBLE_DEVICES=0 python run_dior_det_vllm.py \
  --split test \
#   --limit 20 \
  --batch_size 16 \
  --out dior_pred_vllm.jsonl
