#!/usr/bin/env bash
set -euo pipefail
: "${TASK_ROOT:?}"; : "${DATASET_DIR:?}"; : "${PYTHON_BIN:?}"; : "${TARGET_CONFIG_SOURCE:?}"; : "${TARGET_MODEL_SOURCE:?}"; : "${MTP_TOKENIZER_SOURCE:?}"; : "${MTP_ASSISTANT_SOURCE:?}"
REPO="$TASK_ROOT/repo"; TB="${TENSORBOARD_ROOT:-$TASK_ROOT/tensorboard}"; CHAIN="$TASK_ROOT/chain"; LOGS="$TASK_ROOT/logs"; mkdir -p "$TB" "$CHAIN" "$LOGS" "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
export PYTHONPATH="$REPO/src:$REPO/scripts" HF_HOME="$TASK_ROOT/cache/huggingface" TRANSFORMERS_CACHE="$TASK_ROOT/cache/huggingface" TORCH_HOME="$TASK_ROOT/cache/torch" XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME"
echo "[$(date -Is)] chain start data=$DATASET_DIR" | tee "$LOGS/chain.log"
TARGET_CFG="$CHAIN/target.yaml"
"$PYTHON_BIN" - "$REPO/$TARGET_CONFIG_SOURCE" "$TARGET_CFG" "$DATASET_DIR" "$CHAIN/target" "$TB/target" "$TARGET_MODEL_SOURCE" <<'PY'
import os, sys, yaml
from pathlib import Path
s,t,d,o,tb,m=map(Path,sys.argv[1:]); c=yaml.safe_load(s.read_text()); c.setdefault('run',{}).update(id='full_target_20260809',dataset_dir=str(d),output_dir=str(o)); c.setdefault('model',{}).update(model_id=str(m),model_source=str(m),tokenizer_source=os.environ.get('TARGET_TOKENIZER_SOURCE',str(m))); c.setdefault('training',{}).update(report_to='tensorboard',logging_dir=str(tb),save_steps=500,save_total_limit=10); c['training'].pop('max_steps',None); ge=c.setdefault('golden_eval',{}); ge['enabled']=os.environ.get('GOLDEN_EVAL_ENABLED','0')=='1';
if ge['enabled']:
 ge.update(split_path=os.environ['GOLDEN_SPLIT_PATH'],trigger='evaluate',interval=1,metric_for_best_model='overall_score',greater_is_better=True,save_best_checkpoint=True,output_dir=str(o/'golden_eval'),best_checkpoint_dir=str(o/'best_golden_checkpoint'),weights_config=os.environ.get('GOLDEN_WEIGHTS_CONFIG'),metric_version=os.environ.get('GOLDEN_METRIC_VERSION','v5_4'),metric_log_prefix='golden',required_rows=int(os.environ.get('GOLDEN_REQUIRED_ROWS','100')),max_rows=int(os.environ.get('GOLDEN_MAX_ROWS','100')),max_input_tokens=int(os.environ.get('GOLDEN_MAX_INPUT_TOKENS','4096')),max_new_tokens=int(os.environ.get('GOLDEN_MAX_NEW_TOKENS','1024')),require_unique_rows=False)
c.get('qat',{}).update(schema_path='configs/quantization/gemma4_e2b_mobile_litertlm_schema.yaml'); lm=c.setdefault('lora',{});
lm['dropout']=0.0
lm['target_modules']=r'model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
t.write_text(yaml.safe_dump(c,sort_keys=False))
PY
cd "$REPO"; CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" scripts/train_sft.py --config "$TARGET_CFG" 2>&1 | tee "$LOGS/target_train.log"
ADAPTER="$CHAIN/target/final_adapter"; MERGED="$CHAIN/merged_target"
"$PYTHON_BIN" - "$TARGET_CFG" "$ADAPTER" "$MERGED" <<'PY' 2>&1 | tee "$LOGS/target_merge.log"
import sys,yaml
from pathlib import Path
from ir_training.export.merge_lora import merge_lora_adapter
c=yaml.safe_load(Path(sys.argv[1]).read_text()); m=c['model']; merge_lora_adapter(base_model_id=str(m['model_id']),adapter_dir=sys.argv[2],output_dir=sys.argv[3],model_loader=str(m.get('model_loader','auto_causal_lm')),dtype=str(m.get('dtype','bfloat16')),trust_remote_code=bool(m.get('trust_remote_code',False)),processor_model_id=str(m['model_id']),training_config_path=Path(sys.argv[1]),base_model_source=m.get('model_source'),mobile_training_seed_manifest=m.get('mobile_training_seed_manifest'))
PY
MTPDATA="$CHAIN/mtp_compatible"; mkdir -p "$MTPDATA"
"$PYTHON_BIN" - "$DATASET_DIR" "$MTPDATA" "$MTP_TOKENIZER_SOURCE" <<'PY'
import json,sys
from pathlib import Path
from transformers import AutoTokenizer
d,o,t=map(Path,sys.argv[1:]); tok=AutoTokenizer.from_pretrained(str(t),use_fast=True)
for split in ('train','val'):
 n=0; w=(o/f'{split}.jsonl').open('w',encoding='utf-8')
 for line in (d/f'{split}.jsonl').open(encoding='utf-8'):
  r=json.loads(line); ms=r.get('messages') if isinstance(r,dict) else None
  if not isinstance(ms,list): continue
  x=next((str(a.get('content') or '') for a in reversed(ms) if isinstance(a,dict) and a.get('role')=='assistant'),''); ids=tok(x,add_special_tokens=True).get('input_ids') or []
  if len(ids)>=2: w.write(json.dumps({'input_ids':ids,'labels':ids})+'\n'); n+=1
 w.close(); print(split,n)
PY
MTP_CFG="$CHAIN/mtp.yaml"
"$PYTHON_BIN" - "$REPO/configs/models/gemma4_e2b_mtp_drafter_qat.yaml" "$MTP_CFG" "$MTPDATA" "$MERGED" "$CHAIN/mtp_drafter" "$TB/mtp" "$MTP_ASSISTANT_SOURCE" <<'PY'
import sys,yaml
from pathlib import Path
s,t,d,m,o,tb,a=map(Path,sys.argv[1:]); c=yaml.safe_load(s.read_text()); c['run'].update(id='full_mtp_20260809',dataset_dir=str(d),output_dir=str(o)); c['target'].update(model_id_or_path=str(m),tokenizer_id_or_path=str(a)); c['assistant'].update(model_id_or_path=str(a),best_checkpoint_dir=str(o/'best_checkpoint')); c['training'].pop('max_steps',None); c['training']['tensorboard_log_dir']=str(tb); t.write_text(yaml.safe_dump(c,sort_keys=False))
PY
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" scripts/train_gemma4_mtp_drafter.py --config "$MTP_CFG" --execute --target-model "$MERGED" 2>&1 | tee "$LOGS/mtp_train.log"
G270_CFG="$CHAIN/gemma270m.yaml"
"$PYTHON_BIN" - "$REPO/configs/models/gemma3_270m_ir_qat_sft.yaml" "$G270_CFG" "$DATASET_DIR" "$CHAIN/gemma270m" "$TB/gemma270m" <<'PY'
import sys,yaml
from pathlib import Path
s,t,d,o,tb=map(Path,sys.argv[1:]); c=yaml.safe_load(s.read_text()); c['run'].update(id='full_gemma3_270m_20260809',dataset_dir=str(d),output_dir=str(o)); c['training'].update(report_to='tensorboard',logging_dir=str(tb)); c['training'].pop('max_steps',None); c['golden_eval']['enabled']=False; t.write_text(yaml.safe_dump(c,sort_keys=False))
PY
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}" "$PYTHON_BIN" scripts/train_sft.py --config "$G270_CFG" 2>&1 | tee "$LOGS/gemma270m_train.log"
echo "[$(date -Is)] chain completed" | tee -a "$LOGS/chain.log"
