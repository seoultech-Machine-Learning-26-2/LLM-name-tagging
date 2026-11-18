# C · Qwen3.5-0.8B + LoRA 학습 코드 (Kaggle 노트북 셀 1·2)
#
# - Kaggle 노트북 `yangseorin/llmnametagging`에서 실행한 셀을 그대로 옮겼다 (GPU T4 x2 중 1개 사용).
#   Version 1: V3_rdrop seed 43·44, Version 2: Q4_hier seed 43·44 (두 버전 모두 epoch 3 어댑터·헤드·val 예측 저장).
# - seed 42 실험(Q2, Q3, V1, V2, V3, Q4)은 같은 함수의 Colab 판(드라이브 저장, epoch 1~3 저장)으로 먼저 실행했다.
#   Q2·Q3는 헤드를 `nn.Linear` 하나로 둔 이전 버전으로 학습해 헤드 state_dict 키가 `weight, bias`다
#   (불러올 때 `FlatHead.fc`로 매핑 → experiments/C_ensemble_predict.py `load_head`).
# - 동결 실험(Q1: 동결 임베딩+헤드, Q1b: +질문 중심화)은 Colab에서 별도 코드로 실행했고 그 코드는 보관하지 않았다.
# - `!` 줄은 노트북 셸 명령이다. .py로 실행할 때는 터미널에서 먼저 설치한다.
# - 입력: Kaggle 데이터셋 `dataset`(A의 data/split/train.csv, val.csv). test는 이 파일에서 쓰지 않는다.
# - 출력: /kaggle/working/results/{run}_lora_ep3/, {run}_head_ep3.pt, val_{run}.csv, {run}_metrics.json

# %% 셀 1: 설치·데이터·모델 준비
# !pip uninstall -y -q torchao
# !pip install -q -U transformers peft accelerate flash-linear-attention
import os, glob; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, time, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
D = os.path.dirname(glob.glob('/kaggle/input/**/train.csv', recursive=True)[0]) + '/'
OUT = '/kaggle/working/results/'; os.makedirs(OUT, exist_ok=True)
tr = pd.read_csv(D + 'train.csv'); va = pd.read_csv(D + 'val.csv'); print(D, tr.shape, va.shape)
NAME = 'Qwen/Qwen3.5-0.8B'
tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None: tok.pad_token = tok.eos_token
backbone = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).model.cuda()
TARGET_MIX = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'out_proj']
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
backbone.enable_input_require_grads()
pm = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_MIX))
pm.print_trainable_parameters()   # trainable params: 4,915,200 || all params: 757,308,224 || trainable%: 0.6490
# !nvidia-smi --query-gpu=name,memory.total --format=csv

# %% 셀 2: 평가·헤드·학습 함수
import random, math, json, re
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

CLASSES = sorted(tr.model.unique()); cid = {c: i for i, c in enumerate(CLASSES)}
LABEL_MAP = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
             'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
             'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra',
             'gpt-6-astra','kimi-k3','qwen3.8-max']
assert CLASSES == LABEL_MAP, '순서 불일치!'; print('label_map 순서 일치 ✅')

def company(m):
    for k, v in [('claude','anthropic'),('gemini','google'),('gpt','openai'),('qwen','qwen'),('glm','zai'),('kimi','moonshot')]:
        if m.startswith(k): return v
COMP = np.array([company(c) for c in CLASSES]); COMPS = sorted(set(COMP))
M2C = torch.tensor([COMPS.index(c) for c in COMP], device='cuda')
ytr = tr.model.map(cid).values; yva = va.model.map(cid).values

def hungarian(P, pids):
    pred = P.argmax(1).copy()
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]; r, c = linear_sum_assignment(-np.log(P[idx] + 1e-9)); pred[idx[r]] = c
    return pred

def evaluate(P, y, pids):
    pred = P.argmax(1); cy = np.array([COMPS.index(COMP[i]) for i in y])
    cP = np.stack([P[:, COMP == c].sum(1) for c in COMPS], 1)
    return dict(model_BA=float(balanced_accuracy_score(y, pred)), accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro')),
                company_BA=float(balanced_accuracy_score(cy, cP.argmax(1))),
                company_BA_argmax=float(balanced_accuracy_score(cy, [COMPS.index(COMP[i]) for i in pred])),
                model_BA_hungarian=float(balanced_accuracy_score(y, hungarian(P, pids))))

def save_run(run_id, P, df, metrics, extra):
    out = pd.DataFrame(P, columns=CLASSES); out.insert(0, 'id', df.id.values)
    fname = f'val_{run_id}.csv' if 'seed' in run_id else f'val_{run_id}_seed42.csv'
    out.to_csv(OUT + fname, index=False)
    json.dump({**metrics, **extra}, open(f'{OUT}{run_id}_metrics.json', 'w'), indent=2, default=float)
    print(run_id, {k: round(v, 4) if isinstance(v, float) else v for k, v in {**metrics, **extra}.items() if k != 'history'}, flush=True)

class FlatHead(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 18)
    def forward(s, h): return F.log_softmax(s.fc(h), -1), None

class HierHead(nn.Module):
    def __init__(s, d): super().__init__(); s.comp = nn.Linear(d, len(COMPS)); s.model = nn.Linear(d, 18)
    def forward(s, h):
        lc = F.log_softmax(s.comp(h), -1); z = s.model(h); lm = torch.empty_like(z)
        for c in range(len(COMPS)):
            idx = (M2C == c).nonzero(as_tuple=True)[0]
            lm[:, idx] = F.log_softmax(z[:, idx], -1) + lc[:, c:c+1]
        return lm, lc

def smooth_nll(logp, y, ls):
    return ((1 - ls) * -logp.gather(1, y[:, None]).squeeze(1) + ls * -logp.mean(1)).mean()

def crop(t, rng):
    s = re.split(r'(?<=[.!?])\s+', t.strip())
    if len(s) <= 2: return t
    k = rng.randint(math.ceil(0.5 * len(s)), len(s) + 1); st = rng.randint(0, len(s) - k + 1)
    return ' '.join(s[st:st + k])

def encode(texts):
    return tok(texts, padding=True, truncation=True, max_length=384, pad_to_multiple_of=64, return_tensors='pt').to('cuda')

def pool(enc):
    h = pm(**enc).last_hidden_state.float(); m = enc.attention_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)

def feat(H, center): return torch.cat([H, H - H.mean(0, keepdim=True)], -1) if center else H

@torch.no_grad()
def predict(df, head, center):
    pm.eval(); head.eval(); out = {}
    for _, g in df.groupby('prompt_id'):
        with torch.autocast('cuda', dtype=torch.float16): H = pool(encode(list(g.text)))
        P = head(feat(H, center))[0].exp().cpu().numpy()
        for i, p in zip(g.index, P): out[i] = p
    return np.stack([out[i] for i in df.index])

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_run(run_id, center=True, epochs=3, lr_lora=2e-4, lr_head=1e-3, seed=42,
              ls=0.0, lora_drop=0.05, wd=0.01, crop_aug=False, hier=False, lam=0.5, rdrop=0.0):
    set_seed(seed); torch.cuda.reset_peak_memory_stats()
    for mod in pm.modules():
        if hasattr(mod, 'reset_lora_parameters'): mod.reset_lora_parameters('default', True)
        if hasattr(mod, 'lora_dropout') and 'default' in mod.lora_dropout and isinstance(mod.lora_dropout['default'], nn.Dropout):
            mod.lora_dropout['default'].p = lora_drop
    d = 1024 * (2 if center else 1); head = (HierHead(d) if hier else FlatHead(d)).cuda()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': params, 'lr': lr_lora}, {'params': head.parameters(), 'lr': lr_head}], weight_decay=wd)
    pids = tr.prompt_id.unique(); total = epochs * len(pids); warm = int(0.1 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler(); t0 = time.time(); hist = []

    def step_loss(texts, y):
        with torch.autocast('cuda', dtype=torch.float16): H = pool(encode(texts))
        lm, lc = head(feat(H, center))
        return lm, smooth_nll(lm, y, ls) + (lam * smooth_nll(lc, M2C[y], ls) if hier else 0)

    for ep in range(epochs):
        pm.train(); head.train(); rng = np.random.RandomState(seed + ep); losses = []
        for pid in rng.permutation(pids):
            g = tr[tr.prompt_id == pid]; y = torch.tensor(g.model.map(cid).values, device='cuda')
            texts = [crop(t, rng) for t in g.text] if crop_aug else list(g.text)
            lm1, loss = step_loss(texts, y)
            if rdrop > 0:
                lm2, loss2 = step_loss(texts, y)
                kl = 0.5 * (F.kl_div(lm1, lm2, log_target=True, reduction='batchmean') +
                            F.kl_div(lm2, lm1, log_target=True, reduction='batchmean'))
                loss = 0.5 * (loss + loss2) + rdrop * kl
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        Pva = predict(va, head, center); m = evaluate(Pva, yva, va.prompt_id.values)
        hist.append({'epoch': ep + 1, 'loss': float(np.mean(losses)), **m})
        print(f'[{run_id}] epoch {ep + 1} | loss {np.mean(losses):.3f} | val BA {m["model_BA"]:.4f} | +hung {m["model_BA_hungarian"]:.4f} | {time.time() - t0:.0f}s', flush=True)
    pm.save_pretrained(f'{OUT}{run_id}_lora_ep{epochs}'); torch.save(head.state_dict(), f'{OUT}{run_id}_head_ep{epochs}.pt')
    Ptr = predict(tr, head, center)
    save_run(run_id, Pva, va, m, dict(train_BA=float(balanced_accuracy_score(ytr, Ptr.argmax(1))), sec=round(time.time() - t0), seed=seed,
             peak_GB=round(torch.cuda.max_memory_allocated() / 1e9, 1), center=center, epochs=epochs, lr_lora=lr_lora, lr_head=lr_head,
             ls=ls, lora_drop=lora_drop, wd=wd, crop_aug=crop_aug, hier=hier, lam=lam, rdrop=rdrop, target=TARGET_MIX, r=16, history=hist))
    return head

# %% 실행한 설정 (한 번에 하나씩 실행, 1회 약 1시간 10분 / R-Drop 기준)
# seed 42 (Colab, 같은 함수):
#   train_run('Q2_lora_flat', center=False)                                  # LoRA, 중심화 없음 → val BA 0.328
#   train_run('Q3_lora_centered')                                            # + 질문 중심화      → 0.482
#   train_run('V1_reg', ls=0.1, lora_drop=0.1, wd=0.05)                      # 규제 강화          → 0.472 (미채택)
#   train_run('V2_reg_crop', crop_aug=True, ...)                             # 문장 자르기 증강   → 0.470 (미채택; 나머지 인자는 V2_reg_crop_metrics.json 참고)
#   train_run('V3_rdrop', rdrop=1.0)                                         # R-Drop             → 0.492 (채택)
#   train_run('Q4_hier', rdrop=1.0, hier=True)                               # + 계층형 헤드      → 0.500
# seed 43·44 (Kaggle Version 1·2, 아래 그대로 실행):
for s in [43, 44]:
    train_run(f'V3_rdrop_seed{s}', seed=s, rdrop=1.0)
print('V3 seed 43·44 완료 ✅')
for s in [43, 44]:
    train_run(f'Q4_hier_seed{s}', seed=s, rdrop=1.0, hier=True)
print('Q4 seed 43·44 완료 ✅')
