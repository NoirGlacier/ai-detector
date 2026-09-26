#!/usr/bin/env python3
"""TRAIN BẢN LỚN ĐA NGUỒN + MINILM ENSEMBLE — chạy trên GitHub Actions (4 vCPU / 16GB RAM, free).
Pipeline dữ liệu GIỐNG HỆT train_job.py (cùng seed → cùng tập test) để so sánh công bằng:
  - Model A: TF-IDF + structural → LR  (bản v3, 40.1% top-1)
  - Model B: multilingual-e5-small (ONNX int8, 384 chiều) + structural → LR
  - Ensemble: w * A + (1-w) * B, chọn w trên tập val tách từ train (không nhìn test)
Model B embed qua đúng file ONNX int8 mà web app sẽ dùng lúc serving → không lệch train/serve.
Output: out/detector.joblib + out/metrics.json + out/minilm.onnx + out/minilm_tokenizer.json
"""
import os, sys, json, time, gc, urllib.request

DEPS_OK = True
try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import scipy.sparse as sp
    import joblib
    import onnxruntime
    import transformers
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    DEPS_OK = False

EMB_MODEL = "intfloat/multilingual-e5-small"
EMB_PREFIX = "passage: "
EMB_MAXLEN = 256
HF_BASE = "https://huggingface.co/chiminh652010/ai-detector-v2/resolve/main"

OUT = os.environ.get("OUT_DIR", "out")
os.makedirs(OUT, exist_ok=True)
TMP = "/tmp/data"
os.makedirs(TMP, exist_ok=True)
t0 = time.time()
def log(*a): print(*a, flush=True)
def rss():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) // 1024
    except Exception:
        return -1

# ============ BẢNG ÁNH XẠ (sao chép nguyên vẹn train_job.py) ============
MERGE = {
    'gemini-2.5-pro-preview-05-06': 'gemini-2.5-pro', 'gemini-2.5-pro-preview-03-25': 'gemini-2.5-pro',
    'gemini-2.5-pro-exp-03-25': 'gemini-2.5-pro',
    'gemini-2.5-flash-preview-04-17': 'gemini-2.5-flash',
    'gemini-2.5-flash-lite-preview-06-17-thinking': 'gemini-2.5-flash-lite',
    'gemini-2.0-flash-001': 'gemini-2.0-flash', 'gemini-2.0-flash-thinking-exp-01-21': 'gemini-2.0-flash',
    'grok-3-preview-02-24': 'grok-3', 'grok-3-mini-beta': 'grok-3-mini', 'grok-3-mini-high': 'grok-3-mini',
    'grok-4-0709': 'grok-4', 'grok-4-fast-reasoning': 'grok-4-fast',
    'llama-4-maverick-17b-128e-instruct': 'llama-4-maverick', 'llama-4-maverick-03-26-experimental': 'llama-4-maverick',
    'llama-4-scout-17b-16e-instruct': 'llama-4-scout', 'llama-3.3-70b-instruct': 'llama-3.3-70b',
    'claude-3-7-sonnet-20250219': 'claude-3.7-sonnet', 'claude-3-7-sonnet-20250219-thinking-32k': 'claude-3.7-sonnet',
    'claude-3-5-sonnet-20241022': 'claude-3.5-sonnet', 'claude-3-5-haiku-20241022': 'claude-3.5-haiku',
    'claude-sonnet-4-20250514': 'claude-sonnet-4', 'claude-sonnet-4-20250514-thinking-32k': 'claude-sonnet-4',
    'claude-opus-4-20250514': 'claude-opus-4', 'claude-opus-4-20250514-thinking-16k': 'claude-opus-4',
    'claude-opus-4-1-20250805': 'claude-opus-4.1', 'claude-opus-4-1-20250805-thinking-16k': 'claude-opus-4.1',
    'claude-sonnet-4-5-20250929-old': 'claude-sonnet-4.5', 'claude-sonnet-4-5-20250929-thinking-32k': 'claude-sonnet-4.5',
    'o3-2025-04-16': 'o3', 'o4-mini-2025-04-16': 'o4-mini',
    'gpt-4.1-2025-04-14': 'gpt-4.1', 'gpt-4.1-mini-2025-04-14': 'gpt-4.1-mini',
    'gpt-4o-2024-11-20': 'gpt-4o', 'gpt-4o-mini-2024-07-18': 'gpt-4o-mini',
    'chatgpt-4o-latest-20250326': 'chatgpt-4o-latest', 'chatgpt-4o-latest-20250326-old': 'chatgpt-4o-latest',
    'gpt-5-high': 'gpt-5', 'gpt-5-chat': 'gpt-5', 'gpt-5-high-new-system-prompt': 'gpt-5', 'gpt-5-old': 'gpt-5',
    'gpt-5-mini-high': 'gpt-5-mini',
    'deepseek-v3-0324': 'deepseek-v3', 'deepseek-r1-0528': 'deepseek-r1',
    'deepseek-v3.1': 'deepseek-v3.1', 'deepseek-v3.1-thinking': 'deepseek-v3.1',
    'deepseek-v3.1-terminus': 'deepseek-v3.1', 'deepseek-v3.1-terminus-thinking': 'deepseek-v3.1',
    'qwen3-235b-a22b-instruct-2507': 'qwen3-235b', 'qwen3-235b-a22b': 'qwen3-235b',
    'qwen3-235b-a22b-no-thinking': 'qwen3-235b', 'qwen3-235b-a22b-thinking-2507': 'qwen3-235b',
    'qwen3-235b-a22b-instruct-2507-invalid': 'qwen3-235b',
    'qwen-max-2025-01-25': 'qwen-max', 'qwen3-max-preview': 'qwen3-max', 'qwen3-max-2025-09-23': 'qwen3-max',
    'qwen3-max-2025-09-26': 'qwen3-max',
    'qwen3-coder-480b-a35b-instruct': 'qwen3-coder-480b',
    'qwen3-30b-a3b': 'qwen3-30b-a3b', 'qwen3-30b-a3b-instruct-2507': 'qwen3-30b-a3b',
    'qwen3-next-80b-a3b-instruct': 'qwen3-next-80b', 'qwen3-next-80b-a3b-thinking': 'qwen3-next-80b',
    'kimi-k2-0711-preview': 'kimi-k2', 'kimi-k2-0905-preview': 'kimi-k2',
    'command-a-03-2025': 'command-a', 'hunyuan-turbos-20250416': 'hunyuan-turbos',
    'amazon.nova-pro-v1:0': 'amazon-nova-pro', 'amazon-nova-experimental-chat-05-14': 'amazon-nova-experimental',
    'mistral-small-3.1-24b-instruct-2503': 'mistral-small', 'mistral-small-2506': 'mistral-small',
    'mistral-medium-2505': 'mistral-medium', 'mistral-medium-2508': 'mistral-medium',
    'magistral-medium-2506': 'magistral-medium', 'gemma-3-27b-it': 'gemma-3-27b', 'gemma-3n-e4b-it': 'gemma-3n-e4b',
    'gpt-oss-120b': 'gpt-oss', 'gpt-oss-20b': 'gpt-oss',
    'step-1o-turbo-202506': 'step-1o-turbo', 'mai-1-preview': 'mai-1-preview',
    'longcat-flash-chat': 'longcat-flash', 'glm-4.5': 'glm-4.5', 'glm-4.5-air': 'glm-4.5',
    'glm-4.6': 'glm-4.5', 'glm-4.5v': 'glm-4.5',
    'gemini-2.5-flash-preview-09-2025': 'gemini-2.5-flash',
    'gemini-2.5-flash-lite-preview-09-2025-no-thinking': 'gemini-2.5-flash-lite',
    # === WildChat (nền tảng khác, 2024) ===
    'gpt-4o-2024-04-09': 'gpt-4o', 'gpt-4o-2024-05-13': 'gpt-4o', 'gpt-4o-2024-08-06': 'gpt-4o',
    'gpt-4o-2024-11-20': 'gpt-4o', 'gpt-4-turbo-2024-04-09': 'gpt-4o',
    'chatgpt-4o-latest-2024-05-13': 'chatgpt-4o-latest', 'chatgpt-4o-latest-2024-08-13': 'chatgpt-4o-latest',
    'claude-3-5-sonnet-20240620': 'claude-3.5-sonnet', 'claude-3-haiku-20240307': 'claude-3.5-haiku',
}
FAMILY = {
    'claude': 'Claude (Anthropic)', 'gpt': 'GPT (OpenAI)', 'o3': 'GPT (OpenAI)', 'o4': 'GPT (OpenAI)',
    'chatgpt': 'GPT (OpenAI)', 'gemini': 'Gemini (Google)', 'gemma': 'Gemma (Google)',
    'llama': 'Llama (Meta)', 'qwen': 'Qwen (Alibaba)', 'qwq': 'Qwen (Alibaba)',
    'deepseek': 'DeepSeek', 'grok': 'Grok (xAI)', 'mistral': 'Mistral', 'magistral': 'Mistral',
    'command': 'Mistral (Command A)', 'kimi': 'Kimi (Moonshot)', 'minimax': 'MiniMax',
    'amazon': 'Amazon Nova', 'hunyuan': 'Hunyuan (Tencent)',
    'glm': 'GLM (Z.ai)', 'step': 'StepFun', 'mai': 'MAI (VinAI)', 'longcat': 'LongCat (Meituan)',
}
def family_of(m):
    for k, v in FAMILY.items():
        if m.startswith(k):
            return v
    return 'Khac'
DISPLAY = {
    'gemini-2.5-pro': 'Gemini 2.5 Pro', 'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite (thinking)', 'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'grok-3': 'Grok 3', 'grok-3-mini': 'Grok 3 Mini', 'grok-4': 'Grok 4', 'grok-4-fast': 'Grok 4 Fast',
    'llama-4-maverick': 'Llama 4 Maverick', 'llama-4-scout': 'Llama 4 Scout', 'llama-3.3-70b': 'Llama 3.3 70B',
    'claude-3.7-sonnet': 'Claude 3.7 Sonnet', 'claude-3.5-sonnet': 'Claude 3.5 Sonnet',
    'claude-3.5-haiku': 'Claude 3.5 Haiku', 'claude-sonnet-4': 'Claude Sonnet 4',
    'claude-sonnet-4.5': 'Claude Sonnet 4.5', 'claude-opus-4': 'Claude Opus 4',
    'claude-opus-4.1': 'Claude Opus 4.1', 'o3': 'OpenAI o3', 'o4-mini': 'OpenAI o4-mini',
    'o3-mini': 'OpenAI o3-mini', 'gpt-4.1': 'GPT-4.1', 'gpt-4.1-mini': 'GPT-4.1 mini',
    'gpt-4o': 'GPT-4o', 'gpt-4o-mini': 'GPT-4o mini', 'chatgpt-4o-latest': 'ChatGPT-4o (latest)',
    'gpt-5': 'GPT-5 (high)', 'gpt-5-mini': 'GPT-5 mini', 'gpt-5-nano': 'GPT-5 nano',
    'deepseek-v3': 'DeepSeek V3', 'deepseek-r1': 'DeepSeek R1', 'deepseek-v3.1': 'DeepSeek V3.1',
    'qwen3-235b': 'Qwen3 235B', 'qwen3-30b-a3b': 'Qwen3 30B A3B', 'qwen-max': 'Qwen Max',
    'qwen3-max': 'Qwen3 Max', 'qwen3-coder-480b': 'Qwen3 Coder 480B', 'qwen3-next-80b': 'Qwen3 Next 80B',
    'qwq-32b': 'QwQ 32B', 'kimi-k2': 'Kimi K2', 'command-a': 'Command A',
    'hunyuan-turbos': 'Hunyuan Turbo S', 'amazon-nova-pro': 'Amazon Nova Pro',
    'amazon-nova-experimental': 'Amazon Nova Experimental', 'mistral-small': 'Mistral Small',
    'mistral-medium': 'Mistral Medium', 'magistral-medium': 'Magistral Medium',
    'gemma-3-27b': 'Gemma 3 27B', 'gemma-3n-e4b': 'Gemma 3n E4B', 'minimax-m1': 'MiniMax M1',
    'glm-4.5': 'GLM 4.5', 'gpt-oss': 'GPT-OSS (OpenAI, open weights)',
    'step-1o-turbo': 'Step 1o Turbo (StepFun)', 'mai-1-preview': 'MAI 1 Preview (VinAI)',
    'longcat-flash': 'LongCat Flash',
}

import re
RE_H = [re.compile(r'^#{%d}\s' % n, re.M) for n in range(1, 7)]
RE_UL = re.compile(r'^\s*[\*\-\+]\s', re.M)
RE_OL = re.compile(r'^\s*\d+[\.\)]\s', re.M)
FEAT_NAMES = ['log_len', 'lines', 'nl_ratio', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
              'ul_items', 'ol_items', 'bold', 'italic', 'inline_code', 'code_blocks',
              'links', 'tables_pipes', 'blockquote', 'emoji', 'avg_sent_len',
              'avg_word_len', 'digits_ratio', 'list_density', 'bold_density',
              'uppercase_ratio', 'exclam', 'question_marks', 'h_rule']
def structural_features(text):
    t = text[:8000]
    n = len(t); lines = t.split('\n'); nl = len(lines)
    words = t.split()
    sent = [s for s in re.split(r'[.!?]+', t) if s.strip()]
    emoji = sum(1 for c in t[:4000] if ord(c) > 0x2600 and not c.isalpha())
    alpha = max(1, sum(c.isalpha() for c in t))
    return np.array([
        np.log1p(n), nl, n / max(nl, 1) / 100.0,
        *[len(r.findall(t)) for r in RE_H],
        len(RE_UL.findall(t)), len(RE_OL.findall(t)),
        t.count('**'), t.count('__'),
        t.count('`') - 2 * t.count('```'), t.count('```'),
        len(re.findall(r'\]\(', t)), t.count('|'),
        len(re.findall(r'^\s*>', t, re.M)), emoji,
        np.mean([len(s.split()) for s in sent]) if sent else 0,
        np.mean([len(w) for w in words]) if words else 0,
        sum(c.isdigit() for c in t) / max(n, 1),
        (len(RE_UL.findall(t)) + len(RE_OL.findall(t))) / max(nl, 1),
        t.count('**') / max(nl, 1), sum(c.isupper() for c in t if c.isalpha()) / alpha,
        t.count('!'), t.count('?'),
        len(re.findall(r'^\s*([-*_]\s*){3,}$', t, re.M)),
    ], dtype=np.float32)

def parse_conv_arena(conv):
    """conversation arena: list[{'role','content':[{text}]}] hoac repr-string."""
    if conv is None:
        return ("", "")
    if isinstance(conv, str):
        m1 = re.search(r"'role': 'user'.*?'text': '((?:[^'\\]|\\.)*)'", conv, re.S)
        m2 = re.search(r"'role': 'assistant'.*?'text': '((?:[^'\\]|\\.)*)'", conv, re.S)
        import ast as _ast
        def dec(m):
            if not m:
                return ""
            try:
                return _ast.literal_eval("'" + m.group(1) + "'")
            except Exception:
                return m.group(1)
        return (dec(m1), dec(m2))
    prompt_parts, response = [], None
    for turn in conv:
        try:
            role = turn["role"]
            text = "\n".join(c.get("text") or "" for c in (turn.get("content") or []))
            if role == "user" and response is None:
                prompt_parts.append(text)
            elif role == "assistant" and response is None:
                response = text
        except Exception:
            continue
        if response is not None:
            break
    return ("\n".join(prompt_parts), response or "")

def dl(url, path):
    if not os.path.exists(path):
        log(f"  tải {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)

# ============ BƯỚC 0: XUẤT ONNX INT8 (fail sớm nếu lỗi) ============
def export_embedder(model=None, tokenizer=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tz = tokenizer or AutoTokenizer.from_pretrained(EMB_MODEL)
    d = f"{TMP}/e5_onnx"
    os.makedirs(d, exist_ok=True)
    m = model or AutoModel.from_pretrained(EMB_MODEL)
    m.eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
        def forward(self, input_ids, attention_mask):
            return self.mod(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    wm = _Wrap(m)
    wm.eval()
    enc = tz([EMB_PREFIX + "hello world"], return_tensors="pt", padding=True,
             truncation=True, max_length=EMB_MAXLEN)
    ids, att = enc["input_ids"], enc["attention_mask"]
    kwargs = dict(input_names=["input_ids", "attention_mask"],
                  output_names=["last_hidden_state"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                "attention_mask": {0: "batch", 1: "seq"},
                                "last_hidden_state": {0: "batch", 1: "seq"}},
                  opset_version=18, do_constant_folding=True)
    try:
        with torch.no_grad():
            torch.onnx.export(wm, (ids, att), f"{d}/model.onnx", **kwargs)
    except Exception as e:
        log(f"  torch.onnx.export thường lỗi ({str(e)[:100]}), thử dynamo=True...")
        with torch.no_grad():
            torch.onnx.export(wm, (ids, att), f"{d}/model.onnx", dynamo=True, **kwargs)
    if not os.path.exists(f"{d}/model.onnx"):
        raise RuntimeError("Xuất ONNX thất bại")
    # torch>=2.14 tách weights ra file .data → gộp lại thành 1 file duy nhất
    import onnx
    om = onnx.load(f"{d}/model.onnx")
    for t in om.graph.initializer:
        t.ClearField("external_data")
        t.data_location = onnx.TensorProto.DEFAULT
    onnx.save_model(om, f"{d}/model_full.onnx")
    del om
    for f in ("model.onnx", "model.onnx.data", "model-inferred.onnx"):
        try:
            os.remove(f"{d}/{f}")
        except OSError:
            pass
    log(f"  ONNX fp32 (đã gộp): {os.path.getsize(d + '/model_full.onnx')/1e6:.0f} MB → quantize int8...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(f"{d}/model_full.onnx", f"{OUT}/minilm.onnx", weight_type=QuantType.QInt8)
    os.remove(f"{d}/model_full.onnx")
    tz.backend_tokenizer.save(f"{OUT}/minilm_tokenizer.json")
    log(f"  ✅ minilm.onnx int8: {os.path.getsize(OUT + '/minilm.onnx')/1e6:.0f} MB | tokenizer.json OK")
    return tz

def make_embedder(tz, batch=64):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(f"{OUT}/minilm.onnx", so, providers=["CPUExecutionProvider"])
    names_in = {i.name for i in sess.get_inputs()}
    out_name = None
    for o in sess.get_outputs():
        if len(o.shape) == 3:
            out_name = o.name
            break
    if out_name is None:
        out_name = sess.get_outputs()[0].name
    log(f"  ONNX inputs={sorted(names_in)} output={out_name}")
    def embed(texts):
        outs = []
        for i in range(0, len(texts), batch):
            chunk = [EMB_PREFIX + t[:4000] for t in texts[i:i + batch]]
            enc = tz(chunk, truncation=True, max_length=EMB_MAXLEN, padding=True)
            ids = np.array(enc["input_ids"], dtype=np.int64)
            att = np.array(enc["attention_mask"], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att}
            feed = {k: v for k, v in feed.items() if k in names_in}
            out = sess.run(None, feed)
            h = next(o for o in out if o.ndim == 3)
            mask = att[:, :, None].astype(np.float32)
            emb = (h.astype(np.float32) * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            outs.append(emb)
        return np.vstack(outs)
    return embed

# ============ TF-IDF PIPELINE (GIỐNG HỆT train_job.py) ============
WORD_LEN, CHAR_LEN = 2500, 1200

def fit_tfidf(texts, y):
    rng = np.random.RandomState(42)
    sub = rng.choice(len(texts), size=min(12000, len(texts)), replace=False)
    word_vec = TfidfVectorizer(ngram_range=(1, 3), min_df=15, max_features=60000,
                               sublinear_tf=True, dtype=np.float32)
    word_vec.fit([texts[i][:WORD_LEN] for i in sub])
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=30,
                               max_features=24000, sublinear_tf=True, dtype=np.float32)
    char_vec.fit([texts[i][:CHAR_LEN] for i in sub])
    del sub; gc.collect()
    mats = []
    for i in range(0, len(texts), 4000):
        chunk = texts[i:i + 4000]
        mats.append(sp.hstack([
            word_vec.transform([t[:WORD_LEN] for t in chunk]),
            char_vec.transform([t[:CHAR_LEN] for t in chunk]),
        ]).tocsr())
    X = sp.vstack(mats).tocsr()
    del mats; gc.collect()
    st = np.vstack([structural_features(t) for t in texts])
    scaler = StandardScaler().fit(st)
    X = sp.hstack([X, sp.csr_matrix(scaler.transform(st).astype(np.float32))]).tocsr()
    log(f"    X: {X.shape} nnz={X.nnz/1e6:.1f}M | RAM {rss()}MB")
    clf = LogisticRegression(C=2.0, max_iter=1000, solver='lbfgs')
    clf.fit(X, y)
    del X; gc.collect()
    return word_vec, char_vec, scaler, clf

def tfidf_proba(word_vec, char_vec, scaler, clf, texts):
    preds, probas = [], []
    for i in range(0, len(texts), 1500):
        chunk = texts[i:i + 1500]
        Xt = sp.hstack([
            word_vec.transform([t[:WORD_LEN] for t in chunk]),
            char_vec.transform([t[:CHAR_LEN] for t in chunk]),
            sp.csr_matrix(scaler.transform(np.vstack([structural_features(t) for t in chunk])).astype(np.float32)),
        ]).tocsr()
        preds.extend(clf.predict(Xt))
        probas.append(clf.predict_proba(Xt))
        del Xt
    return np.array(preds), np.vstack(probas)

def full_metrics(proba, classes, y_te, lang_te):
    """classes: list thứ tự cột của proba."""
    pred = np.array(classes)[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")
    top5 = np.argsort(-proba, axis=1)[:, :5]
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_idx[v] for v in y_te])
    top5_acc = float(np.mean([y_idx[i] in top5[i] for i in range(len(y_idx))]))
    fam_of_cls = np.array([family_of(c) for c in classes])
    fam_sets = {f: (fam_of_cls == f) for f in sorted(set(fam_of_cls))}
    fam_true = np.array([family_of(v) for v in y_te])
    fam_pred = np.array([max(fam_sets, key=lambda f: p[fam_sets[f]].sum()) for p in proba])
    fam_acc = float((fam_true == fam_pred).mean())
    out = {"accuracy": float(acc), "macro_f1": float(mf1), "top5_accuracy": top5_acc,
           "family_accuracy": fam_acc}
    msk = lang_te == "en"
    if msk.sum():
        out["en_acc"] = float(accuracy_score(y_te[msk], pred[msk]))
    if (~msk).sum():
        out["other_acc"] = float(accuracy_score(y_te[~msk], pred[~msk]))
    vi = lang_te == "vi"
    if vi.sum() >= 30:
        out["vi_acc"] = float(accuracy_score(y_te[vi], pred[vi]))
    return out

def blend(pa, ca, pb, cb, w):
    """Trộn 2 phân phối xác suất theo tên class. w = trọng số A."""
    bm = {c: j for j, c in enumerate(cb)}
    out = w * pa
    for i, c in enumerate(ca):
        j = bm.get(c)
        if j is not None:
            out[:, i] += (1 - w) * pb[:, j]
    return out

def main():
    # ===== BƯỚC 0: embedder (fail sớm) =====
    log(f"[0] Xuất {EMB_MODEL} → ONNX int8...")
    tz = export_embedder()
    embed = make_embedder(tz)
    # sanity: 2 câu — cặp dịch nghĩa phải gần nhau hơn 2 câu vô liên quan
    e = embed(["Xin chào, tôi có thể giúp gì cho bạn?", "Hello, how can I help you today?",
               "The quadratic formula is negative b plus or minus the square root of b squared minus four a c, all divided by two a."])
    cos = (e @ e.T)
    log(f"  sanity: vi-en={cos[0,1]:.3f} (≥ ~0.6 là ổn), vi-toán={cos[0,2]:.3f} (thấp hơn là đúng)")
    assert np.isfinite(e).all() and abs(np.linalg.norm(e[0]) - 1) < 1e-3, "embedding hỏng!"

    # ===== DỮ LIỆU (giống hệt train_job.py) =====
    rows = {"model": [], "response": [], "language": [], "source": []}
    log("[1/3] arena-human-preference-140k...")
    BASE = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/resolve/main/data/train-%05d-of-00007.parquet"
    for shard in range(7):
        p = f"{TMP}/arena{shard}.parquet"
        dl(BASE % shard, p)
        pf = pq.ParquetFile(p)
        for g in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(g, columns=["model_a", "model_b", "winner",
                                              "conversation_a", "conversation_b",
                                              "language"]).to_pydict()
            for i in range(len(d["winner"])):
                if d["winner"][i] == "tie (botherror)":
                    continue
                lang = d["language"][i] or "und"
                for side in ("a", "b"):
                    if d[f"model_{side}"][i] not in MERGE:
                        continue
                    _, resp = parse_conv_arena(d[f"conversation_{side}"][i])
                    if len(resp) < 40:
                        continue
                    rows["model"].append(MERGE[d[f"model_{side}"][i]])
                    rows["response"].append(resp[:8000])
                    rows["language"].append(lang)
                    rows["source"].append("arena")
        os.remove(p)
        log(f"  shard {shard+1}/7 — tổng {len(rows['model']):,}")

    log("[2/3] arena-expert-5k (11/2025)...")
    p = f"{TMP}/expert.parquet"
    dl("https://huggingface.co/datasets/lmarena-ai/arena-expert-5k/resolve/main/data/train-00000-of-00001.parquet", p)
    pf = pq.ParquetFile(p)
    for g in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(g, columns=["model_a", "model_b", "winner",
                                          "conversation_a", "conversation_b",
                                          "language"]).to_pydict()
        for i in range(len(d["winner"])):
            if d["winner"][i] == "tie (botherror)":
                continue
            lang = d["language"][i] or "und"
            for side in ("a", "b"):
                if d[f"model_{side}"][i] not in MERGE:
                    continue
                _, resp = parse_conv_arena(d[f"conversation_{side}"][i])
                if len(resp) < 40:
                    continue
                rows["model"].append(MERGE[d[f"model_{side}"][i]])
                rows["response"].append(resp[:8000])
                rows["language"].append(lang)
                rows["source"].append("expert")
    log(f"  xong — tổng {len(rows['model']):,}")

    log("[3/3] WildChat-1M...")
    for shard in range(14):
        p = f"{TMP}/wc{shard}.parquet"
        dl(f"https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-{shard:05d}-of-00014.parquet", p)
        pf = pq.ParquetFile(p)
        for g in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(g, columns=["model", "conversation", "language"]).to_pydict()
            for i in range(len(d["model"])):
                if d["model"][i] not in MERGE:
                    continue
                conv = d["conversation"][i]
                response = None
                for turn in (conv or []):
                    if turn.get("role") == "assistant" and response is None:
                        response = turn.get("content") or ""
                    if response is not None:
                        break
                if not response or len(response) < 40:
                    continue
                rows["model"].append(MERGE[d["model"][i]])
                rows["response"].append(response[:8000])
                rows["language"].append(d["language"][i] or "und")
                rows["source"].append("wildchat")
        os.remove(p)
        log(f"  shard {shard+1}/14 — tổng {len(rows['model']):,}")

    df = pd.DataFrame(rows)
    del rows; gc.collect()
    df = df.drop_duplicates(subset=["model", "response"]).reset_index(drop=True)
    vc = df["model"].value_counts()
    df = df[df["model"].isin(vc[vc >= 60].index)]
    parts = []
    for cls, grp in df.groupby("model"):
        src_caps = []
        total = min(len(grp), 2200)
        src_counts = grp["source"].value_counts()
        for src, n in src_counts.items():
            cap = int(total * 0.6)
            src_caps.append(grp[grp.source == src].sample(min(n, cap), random_state=42))
        sel = pd.concat(src_caps)
        if len(sel) > total:
            sel = sel.sample(total, random_state=42)
        parts.append(sel)
    df = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"TỔNG: {len(df):,} mẫu | {df.model.nunique()} class | nguồn: {df.source.value_counts().to_dict()}")
    log(f"RAM {rss()}MB | {time.time()-t0:.0f}s")

    y_all = np.array(df["model"].tolist(), dtype=object)
    lang_all = np.array(df["language"].tolist(), dtype=object)
    resp = df["response"].fillna("").tolist()
    del df; gc.collect()
    idx_tr, idx_te = train_test_split(np.arange(len(resp)), test_size=0.15,
                                      stratify=y_all, random_state=42)
    idx_fit, idx_val = train_test_split(idx_tr, test_size=0.15,
                                        stratify=y_all[idx_tr], random_state=42)
    y_te = y_all[idx_te]; lang_te = lang_all[idx_te]
    y_fit = y_all[idx_fit]; y_val = y_all[idx_val]; y_tr = y_all[idx_tr]
    log(f"Chia: fit={len(idx_fit):,} val={len(idx_val):,} test={len(idx_te):,} (tập test GIỐNG HỆT v3)")

    # ===== EMBED TOÀN BỘ (qua ONNX int8 = đúng đường serving) =====
    log(f"Embed {len(resp):,} văn bản qua minilm.onnx int8 (batch 64, seq {EMB_MAXLEN})...")
    E = np.zeros((len(resp), 384), dtype=np.float32)
    t_emb = time.time()
    for i in range(0, len(resp), 4096):
        E[i:i+4096] = embed(resp[i:i+4096])
        if (i // 4096) % 2 == 0 or i + 4096 >= len(resp):
            rate = (i + 4096 if i + 4096 < len(resp) else len(resp)) / (time.time() - t_emb + 1e-9)
            eta = (len(resp) - min(i + 4096, len(resp))) / max(rate, 1)
            log(f"  {min(i+4096, len(resp)):,}/{len(resp):,} ({rate:.0f} văn bản/s, còn ~{eta/60:.0f} phút)")
    st_all = np.vstack([structural_features(t) for t in resp])
    log(f"Embed xong ({(time.time()-t_emb)/60:.1f} phút) | RAM {rss()}MB")

    def B_matrix(idxs, scaler_b):
        return np.hstack([E[idxs], scaler_b.transform(st_all[idxs]).astype(np.float32)])

    # ===== MODEL A (TF-IDF) — A_val để chọn w, A_full để phục vụ =====
    log("[A] TF-IDF pipeline — fit bản con (chọn w)...")
    resp_fit = [resp[i] for i in idx_fit]
    wv_v, cv_v, sc_v, clf_av = fit_tfidf(resp_fit, y_fit)
    del resp_fit; gc.collect()
    resp_val = [resp[i] for i in idx_val]
    _, pa_val = tfidf_proba(wv_v, cv_v, sc_v, clf_av, resp_val)
    ca = list(clf_av.classes_)
    del resp_val, wv_v, cv_v, sc_v, clf_av; gc.collect()
    log(f"  A_val trên val: {full_metrics(pa_val, ca, y_val, lang_all[idx_val])['accuracy']:.4f}")

    log("[A] TF-IDF pipeline — fit bản đầy đủ (85% train)...")
    resp_tr = [resp[i] for i in idx_tr]
    word_vec, char_vec, scaler, clf = fit_tfidf(resp_tr, y_tr)
    del resp_tr; gc.collect()

    # ===== MODEL B (MiniLM) — sweep C trên val + chọn w ensemble =====
    log("[B] MiniLM pipeline...")
    scaler_bv = StandardScaler().fit(st_all[idx_fit])
    Xb_fit, Xb_val = B_matrix(idx_fit, scaler_bv), B_matrix(idx_val, scaler_bv)
    best = {"acc": 0.0, "C": None, "w": None}
    for C in (1.0, 2.0, 4.0):
        clf_bv = LogisticRegression(C=C, max_iter=1000).fit(Xb_fit, y_fit)
        pb_val = clf_bv.predict_proba(Xb_val)
        cb = list(clf_bv.classes_)
        b_alone = accuracy_score(y_val, np.array(cb)[np.argmax(pb_val, axis=1)])
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            acc = accuracy_score(y_val, np.array(ca)[np.argmax(blend(pa_val, ca, pb_val, cb, w), axis=1)])
            if acc > best["acc"]:
                best = {"acc": acc, "C": C, "w": w}
        log(f"  C={C}: B-alone={b_alone:.4f} | tốt nhất tới giờ: w={best['w']} → {best['acc']:.4f}")
    log(f"  ✅ Chọn: C_b={best['C']}, w_A={best['w']} (val acc {best['acc']:.4f})")
    del Xb_fit, Xb_val; gc.collect()

    # B_full
    scaler_b = StandardScaler().fit(st_all[idx_tr])
    Xb_tr = np.hstack([E[idx_tr], scaler_b.transform(st_all[idx_tr]).astype(np.float32)])
    clf_b = LogisticRegression(C=best["C"], max_iter=1000).fit(Xb_tr, y_tr)
    cb = list(clf_b.classes_)
    del Xb_tr; gc.collect()

    # ===== ĐÁNH GIÁ TRÊN TẬP TEST (bị giữ, chưa từng dùng) =====
    log("[TEST] Đánh giá A / B / ensemble...")
    resp_te = [resp[i] for i in idx_te]
    _, pa = tfidf_proba(word_vec, char_vec, scaler, clf, resp_te)
    del resp_te; gc.collect()
    classes = list(clf.classes_)
    Xb_te = B_matrix(idx_te, scaler_b)
    pb = clf_b.predict_proba(Xb_te)
    pe = blend(pa, classes, pb, cb, best["w"])

    m_a = full_metrics(pa, classes, y_te, lang_te)
    m_b = full_metrics(pb, cb, y_te, lang_te)
    m_e = full_metrics(pe, classes, y_te, lang_te)
    metrics = {**m_e, "n_samples": int(len(y_tr) + len(y_te)), "n_classes": int(len(classes)),
               "w_ensemble": best["w"], "c_b": best["C"], "embedding_model": EMB_MODEL,
               "tfidf_only": m_a, "minilm_only": m_b,
               "val_acc": round(best["acc"], 4)}
    log("\n===== KẾT QUẢ SO SÁNH (cùng tập test) =====")
    log(f"  A (TF-IDF v3)        : {m_a['accuracy']*100:.2f}% | top5 {m_a['top5_accuracy']*100:.1f}% | fam {m_a['family_accuracy']*100:.1f}%")
    log(f"  B (MiniLM e5-small)  : {m_b['accuracy']*100:.2f}% | top5 {m_b['top5_accuracy']*100:.1f}% | fam {m_b['family_accuracy']*100:.1f}%")
    log(f"  ENSEMBLE (w={best['w']}): {m_e['accuracy']*100:.2f}% | top5 {m_e['top5_accuracy']*100:.1f}% | fam {m_e['family_accuracy']*100:.1f}%")
    for k in ("en_acc", "other_acc", "vi_acc"):
        if k in m_e:
            log(f"    {k}: {m_e[k]*100:.2f}%  (A: {m_a.get(k, 0)*100:.2f}% | B: {m_b.get(k, 0)*100:.2f}%)")

    # ===== LƯU =====
    fam_of_cls = np.array([family_of(c) for c in classes])
    fam_sets = {f: list(fam_of_cls == f) for f in sorted(set(fam_of_cls))}
    bundle = {
        'word_vec': word_vec, 'char_vec': char_vec, 'scaler': scaler,
        'feat_names': FEAT_NAMES, 'clf': clf, 'classes': classes,
        'fam_sets': fam_sets,
        'display': {c: DISPLAY.get(c, c) for c in classes},
        'metrics': metrics,
        'word_len': WORD_LEN, 'char_len': CHAR_LEN,
        'minilm': {
            'clf_b': clf_b, 'classes_b': cb, 'scaler_b': scaler_b,
            'w': best['w'], 'max_len': EMB_MAXLEN, 'prefix': EMB_PREFIX,
            'pad_id': int(getattr(tz, 'pad_token_id', 1) or 1),
            'emb_model': EMB_MODEL,
            'url': f"{HF_BASE}/minilm.onnx",
            'tok_url': f"{HF_BASE}/minilm_tokenizer.json",
        },
    }
    out_path = f"{OUT}/detector.joblib"
    joblib.dump(bundle, out_path, compress=5)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False, indent=1)
    log(f"Đã lưu {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=tok)
            repo_id = os.environ.get("HF_REPO", "chiminh652010/ai-detector-v2")
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            for f, rp in ((out_path, "detector.joblib"), (f"{OUT}/metrics.json", "metrics.json"),
                          (f"{OUT}/minilm.onnx", "minilm.onnx"), (f"{OUT}/minilm_tokenizer.json", "minilm_tokenizer.json")):
                api.upload_file(path_or_fileobj=f, path_in_repo=rp, repo_id=repo_id, repo_type="model")
            log(f"✅ ĐÃ PUSH LÊN HUB: https://huggingface.co/{repo_id}")
        except Exception as e:
            log(f"⚠️ Push Hub lỗi (file vẫn còn ở {OUT}/): {e}")
    log(f"HOÀN TẤT sau {(time.time()-t0)/60:.1f} phút | RAM {rss()}MB")

if __name__ == "__main__":
    if not DEPS_OK:
        os.system(f"{sys.executable} -m pip install -q numpy pandas pyarrow scipy scikit-learn joblib huggingface_hub")
        os.system(f"{sys.executable} -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu")
        os.system(f"{sys.executable} -m pip install -q transformers onnx onnxruntime")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
