#!/usr/bin/env python3
"""TRAIN V4 — ĐA NGUỒN + SYNTHETIC 2026 (model mới nhất không có dữ liệu công khai).
Thêm nguồn 4: synthetic_2026.parquet — model 2026 (Gemini 3.8/3.7/3.6, Gemma 4, Qwen3.8...)
trả lời prompt thật lấy từ arena battles, sinh qua API free (văn phong THẬT của từng model).
Nguồn dữ liệu (tăng tính khách quan — 3 nền tảng khác nhau):
  1. lmarena-ai/arena-human-preference-140k  (battle chat, 2024–05/2025)
  2. lmarena-ai/arena-expert-5k              (battle chat, 11/2025 — GPT-5, Opus 4.1...)
  3. allenai/WildChat-1M                     (hội thoại người dùng thật, 2024 — nền tảng khác)
Output: out/detector.joblib + out/metrics.json (+ push lên HF Hub nếu có HF_TOKEN).
"""
import os, sys, json, time, gc, io, glob, urllib.request

DEPS_OK = True
try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import scipy.sparse as sp
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    DEPS_OK = False

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

# ============ BẢNG ÁNH XẠ (tự chứa, không phụ thuộc repo) ============
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
    'glm': 'GLM (Z.ai)', 'step': 'StepFun', 'mimo': 'MiMo (Xiaomi)', 'hy': 'Hunyuan (Tencent)', 'mai': 'MAI (VinAI)', 'longcat': 'LongCat (Meituan)',
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
    # === Synthetic 2026 ===
    'gemini-3.8-flash': 'Gemini 3.8 Flash', 'gemini-3.7-flash': 'Gemini 3.7 Flash',
    'gemini-3.6-flash': 'Gemini 3.6 Flash', 'gemma-4-31b': 'Gemma 4 31B',
    'qwen3.8-27b': 'Qwen3.8 27B', 'glm-5.3': 'GLM 5.3', 'glm-5.3-flash': 'GLM 5.3 Flash',
    'glm-5.2': 'GLM 5.2', 'deepseek-v4.1-flash': 'DeepSeek V4.1 Flash', 'kimi-k3': 'Kimi K3',
    'qwen3.8-max': 'Qwen3.8 Max', 'hy4-preview': 'Hunyuan Hy4 (preview)', 'hy3': 'Hunyuan Hy3',
    'mimo-v2.5-pro': 'MiMo V2.5 Pro', 'minimax-m3': 'MiniMax M3', 'step-3.7-flash': 'Step 3.7 Flash',
    # === 2026 (arena user + wag-bench + excelnt sau này) ===
    'gpt-6-luna': 'GPT-6 Luna', 'gpt-6-sol': 'GPT-6 Sol', 'gpt-6-astra': 'GPT-6 Astra',
    'gpt-6.1-sol': 'GPT-6.1 Sol', 'gpt-5.6-luna': 'GPT-5.6 Luna', 'gpt-5.6-sol': 'GPT-5.6 Sol',
    'gpt-5.6-terra': 'GPT-5.6 Terra', 'gpt-5.5': 'GPT-5.5', 'gpt-5.1': 'GPT-5.1',
    'claude-opus-5': 'Claude Opus 5', 'claude-opus-5.5': 'Claude Opus 5.5',
    'claude-fable-5': 'Claude Fable 5', 'claude-fable-5.1': 'Claude Fable 5.1',
    'claude-sonnet-5': 'Claude Sonnet 5', 'claude-haiku-4-5': 'Claude Haiku 4.5',
    'claude-opus-4-5': 'Claude Opus 4.5', 'claude-opus-4-6': 'Claude Opus 4.6',
    'grok-4.6': 'Grok 4.6', 'grok-4.7': 'Grok 4.7', 'grok-4.20': 'Grok 4.20',
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro', 'glm-5.1': 'GLM 5.1',
    'mimo-v2.6-pro': 'MiMo V2.6 Pro', 'deepseek-v4-pro': 'DeepSeek V4 Pro',
    'deepseek-v4-flash': 'DeepSeek V4 Flash', 'mistral-large-4': 'Mistral Large 4',
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

def parse_conv_wildchat(conv):
    """WildChat: list[{'role','content': str}] -> response đầu tiên."""
    if not conv:
        return ""
    for turn in conv:
        if turn.get("role") == "assistant":
            return turn.get("content") or ""
    return ""

def dl(url, path):
    if not os.path.exists(path):
        log(f"  tải {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)

def main():
    rows = {"model": [], "response": [], "language": [], "source": []}

    # ===== NGUỒN 1: arena-140k =====
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

    # ===== NGUỒN 2: expert-5k =====
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

    # ===== NGUỒN 3: WildChat-1M (người dùng thật, nền tảng khác) =====
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
                # WildChat: [{'role': 'user'/'assistant', 'content': '...'}]
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

    # ===== NGUỒN 4: SYNTHETIC 2026 (model mới, sinh qua API) =====
    log("[4/4] Synthetic 2026 (model mới nhất)...")
    import glob as _glob
    syn_files = _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "synthetic_2026.parquet")) + \
                 _glob.glob("data/synthetic_2026.parquet")
    if syn_files:
        SYNMERGE = {
            'claude-3-7-sonnet': 'claude-3.7-sonnet', 'claude-3-5-sonnet': 'claude-3.5-sonnet',
            'claude-3-5-haiku': 'claude-3.5-haiku', 'qwen3-235b-a22b': 'qwen3-235b',
            'chatgpt-4o': 'chatgpt-4o-latest', 'mistral-small-3.1': 'mistral-small',
            'amazon-nova': 'amazon-nova-experimental', 'gemma-3-27b-it': 'gemma-3-27b',
            'gemma-3n-e4b-it': 'gemma-3n-e4b',
        }
        sdf = pd.read_parquet(syn_files[0], columns=["model", "response", "language", "source"])
        for r in sdf.itertuples(index=False):
            rows["model"].append(SYNMERGE.get(r.model, r.model))
            rows["response"].append(r.response[:8000])
            rows["language"].append(r.language)
            rows["source"].append(r.source)
        log(f"  {len(sdf):,} mẫu synthetic | {sdf.model.value_counts().to_dict()}")
    else:
        log("  (không có file synthetic — train như v3)")

    # ===== GỘP + CÂN BẰNG =====
    df = pd.DataFrame(rows)
    del rows; gc.collect()
    df = df.drop_duplicates(subset=["model", "response"]).reset_index(drop=True)
    vc = df["model"].value_counts()
    df = df[df["model"].isin(vc[vc >= 40].index)]  # v4: 40 để vào cả class 2026 ít mẫu
    # Giới hạn 1 nguồn tối đa 60% mẫu mỗi class (đảm bảo đa dạng nguồn)
    parts = []
    for cls, grp in df.groupby("model"):
        src_caps = []
        total = min(len(grp), 2200)
        src_counts = grp["source"].value_counts()
        for src, n in src_counts.items():
            cap = int(total * 0.6) if len(src_counts) > 1 else total
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
    resp_tr = [resp[i] for i in idx_tr]; resp_te = [resp[i] for i in idx_te]
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]
    lang_te = lang_all[idx_te]
    del resp, y_all, lang_all; gc.collect()

    # ===== TF-IDF (bản ĐẠI: cần ~8-12GB RAM) =====
    WORD_LEN, CHAR_LEN = 2500, 1200
    rng = np.random.RandomState(42)
    sub = rng.choice(len(resp_tr), size=min(12000, len(resp_tr)), replace=False)
    word_vec = TfidfVectorizer(ngram_range=(1, 3), min_df=15, max_features=60000,
                               sublinear_tf=True, dtype=np.float32)
    word_vec.fit([resp_tr[i][:WORD_LEN] for i in sub])
    log(f"Vocab word: {len(word_vec.vocabulary_):,} | RAM {rss()}MB")
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=30,
                               max_features=24000, sublinear_tf=True, dtype=np.float32)
    char_vec.fit([resp_tr[i][:CHAR_LEN] for i in sub])
    log(f"Vocab char: {len(char_vec.vocabulary_):,} | RAM {rss()}MB")
    del sub; gc.collect()

    mats = []
    for i in range(0, len(resp_tr), 4000):
        chunk = resp_tr[i:i + 4000]
        mats.append(sp.hstack([
            word_vec.transform([t[:WORD_LEN] for t in chunk]),
            char_vec.transform([t[:CHAR_LEN] for t in chunk]),
        ]).tocsr())
    X_tr = sp.vstack(mats).tocsr()
    del mats; gc.collect()
    st_tr = np.vstack([structural_features(t) for t in resp_tr])
    scaler = StandardScaler().fit(st_tr)
    X_tr = sp.hstack([X_tr, sp.csr_matrix(scaler.transform(st_tr).astype(np.float32))]).tocsr()
    log(f"X_tr: {X_tr.shape} nnz={X_tr.nnz/1e6:.1f}M | RAM {rss()}MB")

    # ===== LR =====
    log("Fit LR...")
    clf = LogisticRegression(C=2.0, max_iter=1000, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    del X_tr; gc.collect()
    log(f"Fit xong ({time.time()-t0:.0f}s)")

    # ===== ĐÁNH GIÁ =====
    preds, probas = [], []
    for i in range(0, len(resp_te), 1500):
        chunk = resp_te[i:i + 1500]
        Xt = sp.hstack([
            word_vec.transform([t[:WORD_LEN] for t in chunk]),
            char_vec.transform([t[:CHAR_LEN] for t in chunk]),
            sp.csr_matrix(scaler.transform(np.vstack([structural_features(t) for t in chunk])).astype(np.float32)),
        ]).tocsr()
        preds.extend(clf.predict(Xt))
        probas.append(clf.predict_proba(Xt))
        del Xt
    proba = np.vstack(probas)
    pred = np.array(preds)
    classes = clf.classes_
    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")
    top5 = np.argsort(-proba, axis=1)[:, :5]
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_te_idx = np.array([cls_idx[v] for v in y_te])
    top5_acc = float(np.mean([y_te_idx[i] in top5[i] for i in range(len(y_te_idx))]))
    fam_of_cls = np.array([family_of(c) for c in classes])
    fam_sets = {f: (fam_of_cls == f) for f in sorted(set(fam_of_cls))}
    fam_true = np.array([family_of(v) for v in y_te])
    fam_pred = np.array([max(fam_sets, key=lambda f: p[fam_sets[f]].sum()) for p in proba])
    fam_acc = float((fam_true == fam_pred).mean())
    msk = lang_te == "en"
    metrics = {"accuracy": float(acc), "macro_f1": float(mf1), "top5_accuracy": top5_acc,
               "family_accuracy": fam_acc, "n_samples": int(len(y_tr) + len(y_te)),
               "n_classes": int(len(classes))}
    if msk.sum():
        metrics["en_acc"] = float(accuracy_score(y_te[msk], pred[msk]))
    if (~msk).sum():
        metrics["other_acc"] = float(accuracy_score(y_te[~msk], pred[~msk]))
    vi = lang_te == "vi"
    if vi.sum() >= 30:
        metrics["vi_acc"] = float(accuracy_score(y_te[vi], pred[vi]))
    log("\n===== KẾT QUẢ =====")
    for k, v in metrics.items():
        log(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    # ===== LƯU + PUSH HF HUB =====
    fam_sets_ser = {f: list(v) for f, v in fam_sets.items()}
    bundle = {
        'word_vec': word_vec, 'char_vec': char_vec, 'scaler': scaler,
        'feat_names': FEAT_NAMES, 'clf': clf, 'classes': list(classes),
        'fam_sets': fam_sets_ser,
        'display': {c: DISPLAY.get(c, c) for c in classes},
        'metrics': metrics,
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
            api.upload_file(path_or_fileobj=out_path, path_in_repo="detector.joblib",
                            repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=f"{OUT}/metrics.json", path_in_repo="metrics.json",
                            repo_id=repo_id, repo_type="model")
            log(f"✅ ĐÃ PUSH LÊN HUB: https://huggingface.co/{repo_id}")
        except Exception as e:
            log(f"⚠️ Push Hub lỗi (file vẫn còn ở {OUT}/): {e}")
    log(f"HOÀN TẤT sau {(time.time()-t0)/60:.1f} phút | RAM {rss()}MB")

if __name__ == "__main__":
    if not DEPS_OK:
        os.system(f"{sys.executable} -m pip install -q numpy pandas pyarrow scipy scikit-learn joblib huggingface_hub")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
