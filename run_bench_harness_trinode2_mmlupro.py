# -*- coding: utf-8 -*-
"""
Bench-Harness Trinode2 for MMLU-PRO testing.
Calibration: GAOKAO-Bench-2023-2024/Data
Test: MMLU-Pro/results
"""

import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. 工具类：数据对齐与加载 (Data Loader)
# =====================================================================
class GaokaoDataLoader:
    @staticmethod
    def extract_answer(ans):
        if isinstance(ans, list):
            return "".join([str(a).strip().upper() for a in ans])
        return str(ans).strip().upper()

    @staticmethod
    def load_and_align(data_dir: str, models: list, subjects_filter: list = None):
        print(f"\n[*] 正在从 {data_dir} 加载高考先验校准数据...")
        anchor_model = models[0]
        search_pattern = os.path.join(data_dir, f"{anchor_model}_*.json")
        anchor_files = glob.glob(search_pattern)
        
        datasets = {}
        total_aligned = 0
        
        for anchor_file in anchor_files:
            if "seperate" in anchor_file or "Subjective" in anchor_file or "Bench-Harness" in anchor_file or "Objective" in anchor_file: 
                continue
                
            filename = os.path.basename(anchor_file)
            keyword = filename.replace(f"{anchor_model}_", "").replace(".json", "")
            
            if subjects_filter and len(subjects_filter) > 0:
                if not any(sub.lower() in keyword.lower() for sub in subjects_filter):
                    continue
            
            task_data = {}
            valid_dataset = True
            
            for model in models:
                model_file = os.path.join(data_dir, f"{model}_{keyword}.json")
                if not os.path.exists(model_file):
                    valid_dataset = False
                    continue
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        task_data[model] = data.get("example", data.get("examples", data))
                except Exception:
                    valid_dataset = False
                    
            if not valid_dataset: continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            min_len = min(len(task_data[m]) for m in models)
            
            for idx in range(min_len):
                item = base_examples[idx]
                q_text = item.get("question", "").strip()
                if not q_text: continue
                std_ans_str = GaokaoDataLoader.extract_answer(item.get("standard_answer", item.get("answer", [])))
                
                row_corr = []
                row_mod_ans = []
                for model in models:
                    m_ans_raw = task_data[model][idx].get("model_answer", task_data[model][idx].get("answer", []))
                    m_ans_str = GaokaoDataLoader.extract_answer(m_ans_raw)
                    row_corr.append(1 if std_ans_str and std_ans_str == m_ans_str else 0)
                    row_mod_ans.append(m_ans_raw)
                    
                aligned_items.append({
                    "original_item": item,
                    "question": q_text,
                    "std_ans_str": std_ans_str,
                    "model_corr": row_corr,
                    "model_ans": row_mod_ans
                })
                    
            if aligned_items:
                datasets[keyword] = {"items": aligned_items, "anchor_file_path": anchor_file}
                total_aligned += len(aligned_items)
                
        print(f"[+] 加载完毕。成功对齐 {total_aligned} 道题目。")
        return datasets

class MMLUProDataLoader:
    @staticmethod
    def load_and_align(data_dir: str, models: list, subjects_filter: list = None):
        print(f"\n[*] 正在从 {data_dir} 加载 MMLU-PRO 测试数据...")
        anchor_model = models[0]
        anchor_dir = os.path.join(data_dir, anchor_model, "CoT", "all")
        if not os.path.exists(anchor_dir):
            print(f"[!] 致命错误：没有找到基准模型 {anchor_model} 的目录 {anchor_dir}！")
            return {}

        anchor_files = glob.glob(os.path.join(anchor_dir, "*.json"))
        datasets = {}
        total_aligned = 0
        
        for anchor_file in anchor_files:
            filename = os.path.basename(anchor_file)
            keyword = filename.replace(".json", "")
            
            if subjects_filter and len(subjects_filter) > 0:
                allowed = [sub.lower() for sub in subjects_filter]
                if keyword.lower() not in allowed:
                    continue
                    
            task_data = {}
            valid_dataset = True
            
            for model in models:
                model_file = os.path.join(data_dir, model, "CoT", "all", filename)
                if not os.path.exists(model_file):
                    valid_dataset = False
                    continue
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        task_data[model] = json.load(f)
                except Exception:
                    valid_dataset = False
                    
            if not valid_dataset: continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            min_len = min(len(task_data[m]) for m in models)
            
            for idx in range(min_len):
                item = base_examples[idx]
                q_text = item.get("question", "").strip()
                if not q_text: continue
                std_ans_str = str(item.get("answer", "")).strip().upper()
                
                row_corr = []
                row_mod_ans = []
                for model in models:
                    m_pred = str(task_data[model][idx].get("pred", "")).strip().upper()
                    row_corr.append(1 if m_pred == std_ans_str else 0)
                    row_mod_ans.append(m_pred)
                    
                aligned_items.append({
                    "original_item": item,
                    "question": q_text,
                    "std_ans_str": std_ans_str,
                    "model_corr": row_corr,
                    "model_ans": row_mod_ans
                })
                    
            if aligned_items:
                datasets[keyword] = {"items": aligned_items, "anchor_file_path": anchor_file}
                total_aligned += len(aligned_items)
                
        print(f"[+] 加载完毕。成功对齐 {total_aligned} 道题目。")
        return datasets

class BenchHarnessEngine:
    def __init__(self, calib_datasets, args):
        self.args = args
        self.N = len(args.models)
        self.calib_items = []
        for v in calib_datasets.values():
            self.calib_items.extend(v["items"])
            
        self.M = len(self.calib_items)
        if self.M == 0:
            raise ValueError("[❌ 失败] 校准集为空，请检查模型文件是否缺失！")
            
        self.Y_calib = np.array([item["model_corr"] for item in self.calib_items])
        self.K = min(self.args.top_k_fixed, self.M) if self.args.top_k_fixed > 0 else max(5, int(self.M * self.args.top_k_ratio))
        
        self.g_acc = np.mean(self.Y_calib, axis=0)
        self.g_crr = np.zeros((self.N, self.N))
        self.g_fdr = np.zeros((self.N, self.N))
        
        for exec_idx in range(self.N):
            err_mask = (self.Y_calib[:, exec_idx] == 0)
            corr_mask = (self.Y_calib[:, exec_idx] == 1)
            num_err, num_corr = np.sum(err_mask), np.sum(corr_mask)
            for rev_idx in range(self.N):
                if exec_idx == rev_idx: continue
                if num_err > 0: self.g_crr[exec_idx, rev_idx] = np.sum(self.Y_calib[err_mask, rev_idx] == 1) / num_err
                if num_corr > 0: self.g_fdr[exec_idx, rev_idx] = np.sum(self.Y_calib[corr_mask, rev_idx] == 0) / num_corr

        self.architect_idx = int(np.argmax(self.g_acc))
        print(f"\n👑 [系统分配] 经高考校准集排位，全局准确率最高基座为: [{args.models[self.architect_idx]}] (胜率: {self.g_acc[self.architect_idx]*100:.2f}%)")

        # === NEW CODE: Load or Save embeddings ===
        if getattr(self.args, "embed_cache_path", None) and os.path.exists(self.args.embed_cache_path):
            print(f"[*] 检测到指定的本地缓存文件 {self.args.embed_cache_path}，直接加载向量以跳过重复提取!")
            self.V_calib = np.load(self.args.embed_cache_path)
            # Since we loaded from cache, we only need to init the embedder if we route new queries
            print(f"[*] 正在加载向量模型用于推理 (Model: {self.args.embed_model})...")
            self.embedder = SentenceTransformer(self.args.embed_model, device='cpu', model_kwargs={"use_safetensors": False})
        else:
            print(f"[*] 正在加载向量模型 (Model: {self.args.embed_model})...")
            self.embedder = SentenceTransformer(self.args.embed_model, device='cpu', model_kwargs={"use_safetensors": False})
            import hashlib
            cache_key = str(self.M) + "_" + str(self.args.embed_model)
            hasher = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
            # 确保保存到当前工作目录
            cache_file = os.path.join(os.getcwd(), f"embeddings_cache_{hasher}.npy")
            
            if os.path.exists(cache_file):
                print(f"[*] 发现本地缓存文件 {cache_file}，直接加载向量以跳过重复提取!")
                self.V_calib = np.load(cache_file)
            else:
                print(f"[*] 未发现本地缓存，开始提取特征向量并保存...")
                calib_questions = [item["question"] for item in self.calib_items]
                self.V_calib = self.embedder.encode(calib_questions, show_progress_bar=True)
                np.save(cache_file, self.V_calib)
                print(f"[*] 提取完毕, 向量已成功保存至 {cache_file}!")

    def route(self, q_new_text: str):
        q_new_emb = self.embedder.encode([q_new_text], show_progress_bar=False)[0]
        sims = cosine_similarity(q_new_emb.reshape(1, -1), self.V_calib)[0]
        top_k_indices = np.argsort(sims)[-self.K:]
        
        Y_local = self.Y_calib[top_k_indices]
        local_scores = np.sum(Y_local, axis=0)
        exec_idx = int(np.argmax(local_scores))
        exec_local_acc = local_scores[exec_idx] / self.K
        
        err_mask = (Y_local[:, exec_idx] == 0)
        corr_mask = (Y_local[:, exec_idx] == 1)
        num_err, num_corr = np.sum(err_mask), np.sum(corr_mask)
        
        best_rev_idx, best_obj_score = -1, -float('inf')
        lam, eps = self.args.lambda_smooth, 1e-9 
        
        for j in range(self.N):
            if j == exec_idx: continue
            crr_hat = (np.sum(Y_local[err_mask, j] == 1) + lam * self.g_crr[exec_idx, j]) / (num_err + lam + eps)
            fdr_hat = (np.sum(Y_local[corr_mask, j] == 0) + lam * self.g_fdr[exec_idx, j]) / (num_corr + lam + eps)
            acc_hat = (np.sum(Y_local[:, j] == 1) + lam * self.g_acc[j]) / (self.K + lam + eps)
            
            obj_score = self.args.alpha * crr_hat - self.args.beta * fdr_hat + self.args.gamma * acc_hat
            if obj_score > best_obj_score:
                best_obj_score = obj_score
                best_rev_idx = j
                
        if best_rev_idx == -1: 
            best_rev_idx = (exec_idx + 1) % self.N
            
        return exec_idx, best_rev_idx, exec_local_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--test_dir", type=str, default="./MMLU-Pro/results")
    parser.add_argument("--output_dir", type=str, default="./MMLU-Pro/results_trinode")
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--embed_cache_path", type=str, default=None, help="本地向量缓存的绝 对或相对路径")
    parser.add_argument("--top_k_ratio", type=float, default=0.30)
    parser.add_argument("--top_k_fixed", type=int, default=-1)
    parser.add_argument("--lambda_smooth", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--early_exit_threshold", type=float, default=0.85)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("🚀 Bench-Harness: 高考作先验 -> 测试 MMLU-PRO")
    print("=" * 100)

    calib_datasets = GaokaoDataLoader.load_and_align(args.calib_dir, args.models)
    if not calib_datasets: return
        
    engine = BenchHarnessEngine(calib_datasets, args)
    architect_idx = engine.architect_idx
    architect_model = args.models[architect_idx]
    
    test_datasets = MMLUProDataLoader.load_and_align(args.test_dir, args.models, args.subjects)
    if not test_datasets: return
    
    subject_metrics = {}
    global_total_q = 0
    
    for keyword, data in test_datasets.items():
        items = data["items"]
        subject_metrics[keyword] = {
            "total": 0, "experts": np.zeros(len(args.models)),
            "bh_exec": 0, "bh_rev": 0, "bh_final": 0, 
            "rescue": 0, "early_exit": 0, "escalation": 0
        }
        
        for item in tqdm(items, desc=f"Evaluating [{keyword[:25]:<27}]"):
            global_total_q += 1
            subject_metrics[keyword]["total"] += 1
            std_ans_str = item["std_ans_str"]
            q_text = item["question"]
            
            for j in range(len(args.models)):
                subject_metrics[keyword]["experts"][j] += item["model_corr"][j]
                
            exec_idx, rev_idx, exec_local_acc = engine.route(q_text)
            
            exec_corr = item["model_corr"][exec_idx]
            rev_corr = item["model_corr"][rev_idx] 
            subject_metrics[keyword]["bh_exec"] += exec_corr
            subject_metrics[keyword]["bh_rev"] += rev_corr
            
            exec_ans_str = str(item["model_ans"][exec_idx])
            rev_ans_str = str(item["model_ans"][rev_idx])
            arch_ans_str = str(item["model_ans"][architect_idx])
            
            if exec_local_acc >= args.early_exit_threshold:
                final_ans_str = exec_ans_str
                subject_metrics[keyword]["early_exit"] += 1
            else:
                if exec_ans_str != rev_ans_str:
                    final_ans_str = arch_ans_str
                    subject_metrics[keyword]["escalation"] += 1
                else:
                    final_ans_str = exec_ans_str
                    
            final_corr = 1 if std_ans_str and std_ans_str == final_ans_str else 0
            subject_metrics[keyword]["bh_final"] += final_corr
            
            if exec_corr == 0 and final_corr == 1:
                subject_metrics[keyword]["rescue"] += 1

    valid_subjects = [s for s in sorted(subject_metrics.keys()) if subject_metrics[s]["total"] > 0]
    
    col_w_first = 28
    col_w = 14
    header_str = f"| {'Model / Metric':<{col_w_first}} | " + " | ".join([f"{s[:10]+'..':<{col_w}}" for s in valid_subjects]) + f" | {'Average':<11} |"
    sep_str = "| " + "-" * col_w_first + " | " + " | ".join(["-" * col_w for _ in valid_subjects]) + " | " + "-" * 11 + " |"
    
    print("\n\n" + "=" * len(header_str))
    print(header_str)
    print(sep_str)
    
    total_qs = sum(subject_metrics[s]["total"] for s in valid_subjects)
    qs_row = f"| {'Qs (Count)':<{col_w_first}} | " + " | ".join([f"{subject_metrics[s]['total']:<{col_w}}" for s in valid_subjects]) + f" | {total_qs:<11} |"
    print(qs_row)
    print(sep_str)
    
    model_stats = []
    for j, model in enumerate(args.models):
        row_vals = []
        weighted_sum = 0
        for s in valid_subjects:
            q_c = subject_metrics[s]["total"]
            acc = (subject_metrics[s]["experts"][j] / q_c) * 100
            weighted_sum += subject_metrics[s]["experts"][j]
            row_vals.append(f"{acc:.2f}%")
        avg = (weighted_sum / total_qs) * 100 if total_qs else 0
        model_stats.append((avg, model, row_vals, j))
        
    model_stats.sort(key=lambda x: x[0])
    for avg, model, row_vals, j in model_stats:
        m_name = model
        if j == engine.architect_idx: m_name = "👑 " + m_name
        import unicodedata
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in m_name)
        padding = max(0, col_w_first - display_width)
        row_str = f"| {m_name}{' ' * padding} | " + " | ".join([f"{v:<{col_w}}" for v in row_vals]) + f" | {avg:.2f}%{' ':>4} |"
        print(row_str)
        
    print(sep_str)
    
    def print_bh_row(name, key, is_pct=True, is_gain=False):
        row_vals = []
        weighted_sum = 0
        for s in valid_subjects:
            q_c = subject_metrics[s]["total"]
            if is_gain:
                bh_f = (subject_metrics[s]["bh_final"] / q_c) * 100
                best_ex = max([(subject_metrics[s]["experts"][idx] / q_c) * 100 for idx in range(len(args.models))])
                gain = bh_f - best_ex
                weighted_sum += gain * q_c
                sign = "+" if gain > 0 else ""
                row_vals.append(f"{sign}{gain:.2f}%")
            elif is_pct:
                acc = (subject_metrics[s][key] / q_c) * 100
                weighted_sum += subject_metrics[s][key]
                row_vals.append(f"{acc:.2f}%")
            else:
                val = subject_metrics[s][key]
                weighted_sum += val
                row_vals.append(f"{val}")
        if is_gain:
            avg = weighted_sum / total_qs
            sign = "+" if avg > 0 else ""
            avg_str = f"{sign}{avg:.2f}%"
        elif is_pct:
            avg = (weighted_sum / total_qs) * 100 if total_qs else 0
            avg_str = f"{avg:.2f}%"
        else:
            avg_str = f"{int(weighted_sum)}"
            
        import unicodedata
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in name)
        padding = max(0, col_w_first - display_width)
        row_str = f"| {name}{' ' * padding} | " + " | ".join([f"{v:<{col_w}}" for v in row_vals]) + f" | {avg_str:<11} |"
        print(row_str)
        
    print_bh_row("BH-Exec", "bh_exec")
    print_bh_row("BH-Rev", "bh_rev") 
    print_bh_row("BH-Final", "bh_final")
    print_bh_row("Gain (vs Best Exp)", "gain", is_gain=True)
    print_bh_row("Early-Exit", "early_exit", is_pct=False)
    print_bh_row("Escalation", "escalation", is_pct=False)

if __name__ == "__main__":
    main()
