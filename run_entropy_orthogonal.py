# -*- coding: utf-8 -*-
"""
Bench-Harness V2.0: 认知拓扑引擎 (Cognitive Topology Engine)
🌟 核心创新一：认知熵波形特征与 DTW 流形路由 (Cognitive Entropy Waveform & DTW Routing)
🌟 核心创新二：动态正交法庭与 Gram 行列式体积推举 (Geometric Orthogonal Tribunal)
"""

import os
import json
import glob
import zlib
import argparse
import numpy as np
import unicodedata
from tqdm import tqdm

# 尝试探测深度学习环境
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =====================================================================
# 🌟 创新一组件：认知熵探针 (Cognitive Entropy Probe)
# =====================================================================
class EntropyWaveformProbe:
    """
    抛弃文本语义相似度！
    利用滑动窗口计算 Kolmogorov 复杂度（zlib 压缩率），或真实的轻量 LLM 概率分布香农熵。
    将其映射为固定维度的“认知波形 (Cognitive Waveform)”，用于后续的 DTW 时序对齐。
    """
    def __init__(self, use_neural=False, model_name="models/Qwen2.5-0.5B-Instruct", target_dim=32):
        self.target_dim = target_dim
        self.use_neural = use_neural and HAS_TORCH
        
        if self.use_neural:
            print(f"[*] 🧠 神经探针已激活：正在加载真实模型 [{model_name}] 提取 Token 概率香农熵...")
            import warnings
            warnings.filterwarnings("ignore")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
        else:
            print("[*] ⚡ 降级探针已激活：利用信息论 LZ77 算法 (zlib 滑动窗口) 极速提取文本信息密度波形...")

    def extract_waveform(self, text: str) -> np.ndarray:
        if self.use_neural:
            # 【真实模式】捕获语言模型在阅读时的“脑电波”
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, :-1, :]
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).cpu().numpy()
            wave = entropy
        else:
            # 【信息论模式】基于 Kolmogorov 复杂性的完美无模型替代方案
            window_size = max(10, len(text) // self.target_dim)
            if len(text) < window_size:
                text = text.ljust(window_size, " ")
            wave = []
            text_bytes = text.encode('utf-8')
            for i in range(len(text_bytes) - window_size + 1):
                chunk = text_bytes[i:i+window_size]
                compressed = zlib.compress(chunk)
                wave.append(len(compressed) / len(chunk))
            wave = np.array(wave)

        if len(wave) == 0:
            return np.zeros(self.target_dim)

        # 降采样平滑 (Down-pooling)，将任意长度波形映射到定长 32 维平面
        x_old = np.linspace(0, 1, len(wave))
        x_new = np.linspace(0, 1, self.target_dim)
        pooled_wave = np.interp(x_new, x_old, wave)
        
        # Z-Score 归一化
        std = np.std(pooled_wave)
        if std > 1e-9: pooled_wave = (pooled_wave - np.mean(pooled_wave)) / std
        else: pooled_wave = pooled_wave - np.mean(pooled_wave)
            
        return pooled_wave

def compute_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
    """动态时间规整(DTW)：滚动数组空间优化的 O(N*M) 时序比对算法"""
    r, c = len(s1), len(s2)
    d0 = np.full(c + 1, np.inf)
    d0[0] = 0
    for i in range(r):
        d1 = np.full(c + 1, np.inf)
        for j in range(c):
            cost = abs(s1[i] - s2[j])
            d1[j + 1] = cost + min(d0[j], d0[j + 1], d1[j])
        d0 = d1
    return float(d0[-1])

# =====================================================================
# 工具类：数据加载对齐 
# =====================================================================
class GaokaoDataLoader:
    @staticmethod
    def extract_answer(ans):
        if isinstance(ans, list): return "".join([str(a).strip().upper() for a in ans])
        return str(ans).strip().upper()

    @staticmethod
    def load_and_align(data_dir: str, models: list, subjects_filter: list = None):
        anchor_model = models[0]
        anchor_files = glob.glob(os.path.join(data_dir, f"{anchor_model}_*.json"))
        datasets = {}
        for anchor_file in anchor_files:
            if any(k in anchor_file for k in ["seperate", "Subjective", "Bench-Harness", "Objective"]): continue
            keyword = os.path.basename(anchor_file).replace(f"{anchor_model}_", "").replace(".json", "")
            
            if subjects_filter and len(subjects_filter) > 0:
                if not any(sub.lower() in keyword.lower() for sub in subjects_filter):
                    continue

            task_data, valid = {}, True
            for model in models:
                model_file = os.path.join(data_dir, f"{model}_{keyword}.json")
                if not os.path.exists(model_file): valid = False; break
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            task_data[model] = data.get("example", data.get("examples", data))
                        else:
                            task_data[model] = data
                except Exception:
                    valid = False
            if not valid: continue
            
            aligned_items = []
            min_len = min(len(task_data[m]) for m in models)
            for idx in range(min_len):
                item = task_data[anchor_model][idx]
                if not isinstance(item, dict): continue
                q_text = item.get("question", "").strip()
                if not q_text: continue
                std_ans_str = GaokaoDataLoader.extract_answer(item.get("standard_answer", item.get("answer", [])))
                row_corr, row_mod_ans = [], []
                for m in models:
                    m_item = task_data[m][idx]
                    m_ans_raw = m_item.get("model_answer", m_item.get("answer", [])) if isinstance(m_item, dict) else m_item
                    m_ans_str = GaokaoDataLoader.extract_answer(m_ans_raw)
                    row_corr.append(1 if std_ans_str and std_ans_str == m_ans_str else 0)
                    row_mod_ans.append(m_ans_raw)
                aligned_items.append({
                    "original_item": item, "question": q_text, 
                    "std_ans_str": std_ans_str, "model_corr": row_corr, "model_ans": row_mod_ans
                })
            if aligned_items: datasets[keyword] = {"anchor_file_path": anchor_file, "items": aligned_items}
        return datasets

# =====================================================================
# 🌟 核心引擎：认知流形特征与高维正交法庭
# =====================================================================
class EntropyOrthogonalEngine:
    def __init__(self, calib_datasets, args):
        self.args = args
        self.N = len(args.models)
        self.calib_items = []
        for v in calib_datasets.values(): self.calib_items.extend(v["items"])
        self.M = len(self.calib_items)
        if self.M == 0: raise ValueError("[❌] 校准集为空，请检查模型文件或所选学科是否包含有效数据！")
        
        self.Y_calib = np.array([item["model_corr"] for item in self.calib_items])
        
        # 🌟 修改点：使用比例动态计算 K，并保证最少有 3 题来张成 Gram 行列式多面体 (防秩亏坍缩)
        self.K = min(self.M, max(3, int(self.M * self.args.top_k_ratio)))
        
        # 1. 计算宏观正交先验与默认最高架构师
        self.g_acc = np.mean(self.Y_calib, axis=0)
        self.global_architect_idx = int(np.argmax(self.g_acc))
        self.g_crr, self.g_fdr = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        
        for e_idx in range(self.N):
            err_mask, corr_mask = (self.Y_calib[:, e_idx] == 0), (self.Y_calib[:, e_idx] == 1)
            for r_idx in range(self.N):
                if e_idx == r_idx: continue
                if np.sum(err_mask) > 0: self.g_crr[e_idx, r_idx] = np.sum(self.Y_calib[err_mask, r_idx] == 1) / np.sum(err_mask)
                if np.sum(corr_mask) > 0: self.g_fdr[e_idx, r_idx] = np.sum(self.Y_calib[corr_mask, r_idx] == 0) / np.sum(corr_mask)

        # 2. 初始化探针并给校准集打上“波形拓扑钢印”
        self.probe = EntropyWaveformProbe(use_neural=args.use_neural_probe)
        print(f"\n[*] 🌊 正在为 {self.M} 道历史考题进行【离线认知波形拓扑扫描】(Top-K Ratio: {self.args.top_k_ratio*100}%, 实际检索空间 K={self.K})...")
        self.W_calib = np.array([self.probe.extract_waveform(item["question"]) for item in tqdm(self.calib_items, desc="Offline DTW Profiling")])

    def route_nodes(self, q_new_text: str):
        # 🌟 创新一：提取新题波形，使用 DTW 检索出“认知坑点波形最相似”的 K 道题
        q_wave = self.probe.extract_waveform(q_new_text)
        dtw_distances = np.array([compute_dtw(q_wave, w) for w in self.W_calib])
        top_k_indices = np.argsort(dtw_distances)[:self.K]
        
        Y_local = self.Y_calib[top_k_indices]
        local_scores = np.sum(Y_local, axis=0)
        
        # 提取执行者
        exec_idx = int(np.argmax(local_scores))
        exec_local_acc = local_scores[exec_idx] / self.K
        
        # 提取互补审查者
        err_mask, corr_mask = (Y_local[:, exec_idx] == 0), (Y_local[:, exec_idx] == 1)
        best_rev_idx, best_obj_score = -1, -float('inf')
        lam, eps = self.args.lambda_smooth, 1e-9 
        for j in range(self.N):
            if j == exec_idx: continue
            crr_hat = (np.sum(Y_local[err_mask, j] == 1) + lam * self.g_crr[exec_idx, j]) / (np.sum(err_mask) + lam + eps)
            fdr_hat = (np.sum(Y_local[corr_mask, j] == 0) + lam * self.g_fdr[exec_idx, j]) / (np.sum(corr_mask) + lam + eps)
            acc_hat = (np.sum(Y_local[:, j] == 1) + lam * self.g_acc[j]) / (self.K + lam + eps)
            
            obj_score = self.args.alpha * crr_hat - self.args.beta * fdr_hat + self.args.gamma * acc_hat
            if obj_score > best_obj_score: best_obj_score, best_rev_idx = obj_score, j
                
        if best_rev_idx == -1: best_rev_idx = (exec_idx + 1) % self.N
        return exec_idx, best_rev_idx, exec_local_acc, Y_local

    def elect_geometric_tribunal(self, Y_local: np.ndarray, exec_idx: int, rev_idx: int):
        """
        🌟 创新二：Geometric Orthogonal Tribunal (正交法庭)
        解算特征矩阵的 Gram 行列式，推举与前线瞎猜的两人【认知正交性最强、空间盲区绝对互补】的降维裁判！
        """
        if self.K < 3 or self.N < 3: 
            return self.global_architect_idx, 0.0, False 
        
        E_vec = Y_local[:, exec_idx].astype(float)
        R_vec = Y_local[:, rev_idx].astype(float)
        
        eps_noise = 1e-4
        E_vec += np.random.normal(0, eps_noise, self.K)
        R_vec += np.random.normal(0, eps_noise, self.K)
        
        best_arch_idx = self.global_architect_idx
        max_vol_sq = 0.0
        
        for i in range(self.N):
            if i in (exec_idx, rev_idx): continue
            A_vec = Y_local[:, i].astype(float) + np.random.normal(0, eps_noise, self.K)
            
            # 构建认知多面体矩阵，计算 Gram 行列式体积
            V = np.column_stack([E_vec, R_vec, A_vec])
            vol_sq = np.linalg.det(V.T @ V)
            
            if vol_sq > max_vol_sq:
                max_vol_sq = vol_sq
                best_arch_idx = i
                
        is_orthogonal = max_vol_sq > (eps_noise * 10)
        if not is_orthogonal: best_arch_idx = self.global_architect_idx
            
        return best_arch_idx, max_vol_sq, is_orthogonal

# =====================================================================
# 主状态机流转入口 (包含全图排版矩阵打印)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Bench-Harness V2")
    parser.add_argument("--calib_dir", type=str, default="./GAOKAO-Bench-2023-2024/Data")
    parser.add_argument("--test_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--output_dir", type=str, default="./Data_V2_Output")
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="指定评测学科 (不传则默认全部)")
    
    # 🌟 V2 新增/修改参数
    parser.add_argument("--use_neural_probe", action="store_true", help="若开启，自动加载微型 LLM 计算神经波形")
    
    # 👇 修改此处，使用 ratio 取代绝对数量 👇
    parser.add_argument("--top_k_ratio", type=float, default=0.3, help="DTW 匹配时的近邻题数比例 (例如 0.3 代表 30%)")
    
    parser.add_argument("--early_exit_threshold", type=float, default=0.7)
    parser.add_argument("--lambda_smooth", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    print("=" * 140)
    print(f"{'🚀 Bench-Harness V2.0 学科突破版：[创新一] 熵波形DTW路由 + [创新二] 几何正交法庭 🚀':^135}")
    print("=" * 140)
    
    calib_datasets = GaokaoDataLoader.load_and_align(args.calib_dir, args.models, args.subjects)
    if not calib_datasets: return
    
    engine = EntropyOrthogonalEngine(calib_datasets, args)
    test_datasets = GaokaoDataLoader.load_and_align(args.test_dir, args.models, args.subjects)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 🌟 新增各学科状态追踪
    subject_metrics = {}

    print(f"\n🏃‍♂️ 正在进入高维数学观测域：进行免交互跨域仿真...")
    
    for keyword, data in test_datasets.items():
        bh_output_examples = []
        subject_metrics[keyword] = {
            "total": 0, 
            "experts": np.zeros(len(args.models)),
            "bh_exec": 0, 
            "bh_final": 0, 
            "early_exit": 0, 
            "orthogonal_rescue": 0, 
            "fallback_rescue": 0,
            "rescue": 0
        }
        for item in tqdm(data["items"], desc=f"Evaluating [{keyword[:30]}]"):
            subject_metrics[keyword]["total"] += 1
            std_ans = item["std_ans_str"]
            
            # 记录各个基座的答题正确与否
            for j in range(len(args.models)):
                subject_metrics[keyword]["experts"][j] += item["model_corr"][j]
            
            # 🌟 触发创新一：基于波形的 DTW 流形映射
            exec_idx, rev_idx, exec_local_acc, Y_local = engine.route_nodes(item["question"])
            
            # 第一防线执行者的命中记录
            exec_corr = item["model_corr"][exec_idx]
            subject_metrics[keyword]["bh_exec"] += exec_corr
            
            exec_ans = GaokaoDataLoader.extract_answer(item["model_ans"][exec_idx])
            rev_ans = GaokaoDataLoader.extract_answer(item["model_ans"][rev_idx])
            
            action_log, arch_info = "", ""
            
            # 🌟 核心状态机控制流
            if exec_local_acc >= args.early_exit_threshold:
                # [状态 1]: 自信锁早退
                final_ans_raw = item["model_ans"][exec_idx]
                subject_metrics[keyword]["early_exit"] += 1
                action_log = f"🔒【DTW 波形早退】检测到该拓扑域处于认知顺境。指派专才 {args.models[exec_idx]} 局部胜率极高，越过审查阶段！"
            else:
                if exec_ans == rev_ans:
                    # [状态 2]: 艰难共识
                    final_ans_raw = item["model_ans"][exec_idx]
                    action_log = f"🤝【达成共识】红蓝节点同时进入认知波形陷阱域，但意见巧合统一，维持原判。"
                else:
                    # 💥 [状态 3]: 创新二触发！高维正交裁判
                    arch_idx, vol_sq, is_orthogonal = engine.elect_geometric_tribunal(Y_local, exec_idx, rev_idx)
                    final_ans_raw = item["model_ans"][arch_idx]
                    arch_model = args.models[arch_idx]
                    arch_info = f"3. 节点C (正交法庭: {arch_model} | 计算体积 Vol={vol_sq:.4e}) 裁决卷宗: {final_ans_raw}\n"
                    
                    if is_orthogonal:
                        subject_metrics[keyword]["orthogonal_rescue"] += 1
                        action_log = f"⚖️【Geometric Tribunal (正交法庭接管)】分歧爆发！计算 Gram 行列式推举认知盲区【最不重合】的降维裁判 [{arch_model}] 定夺局势！"
                    else:
                        subject_metrics[keyword]["fallback_rescue"] += 1
                        action_log = f"🛡️【Fallback (威权兜底)】分歧爆发！但数学空间完全坍缩(无互斥可言)，迫降唤醒全局大基座 [{arch_model}]。"

            final_ans_str = GaokaoDataLoader.extract_answer(final_ans_raw)
            final_corr = 1 if std_ans == final_ans_str else 0
            subject_metrics[keyword]["bh_final"] += final_corr
            
            if exec_corr == 0 and final_corr == 1:
                subject_metrics[keyword]["rescue"] += 1
            
            # 写入极具科幻感的验证日志
            ex_out = item["original_item"].copy()
            ex_out["model_answer"] = final_ans_raw if isinstance(final_ans_raw, list) else [final_ans_raw]
            ex_out["model_output"] = (
                f"【Bench-Harness V2 高维引擎诊断仪】\n"
                f"1. 节点A(Exec): {args.models[exec_idx]} (当前波形域局部胜率 {exec_local_acc*100:.1f}%)\n"
                f"2. 节点B(Rev):  {args.models[rev_idx] if exec_local_acc < args.early_exit_threshold else '(-)'}\n"
                f"{arch_info}"
                f"4. 状态机动作:\n   -> {action_log}\n"
            )
            bh_output_examples.append(ex_out)
            
        # JSON 覆写
        save_path = os.path.join(args.output_dir, f"Bench-Harness-V2_{keyword}.json")
        with open(data["anchor_file_path"], 'r', encoding='utf-8') as f: p_data = json.load(f)
        p_data["model_name"] = "Bench-Harness-V2"
        p_data["example" if "example" in p_data else "examples"] = bh_output_examples
        with open(save_path, 'w', encoding='utf-8') as f: json.dump(p_data, f, ensure_ascii=False, indent=4)

    # =====================================================================
    # 4. 横向超宽矩阵排版打印 (完美适配您的命令行输出格式)
    # =====================================================================
    sorted_subjects = sorted(subject_metrics.keys())
    valid_subjects = [subj for subj in sorted_subjects if subject_metrics[subj]["total"] > 0]
    
    col_w_first = 35 # 稍微加宽，以便容纳长模型名
    col_w = 14
    header_str = f"| {'Model / Metric':<{col_w_first}} | " + " | ".join([f"{s.replace('2010-2022_', '').replace('2010-2013_', '').replace('2012-2022_', '')[:10]+'..':<{col_w}}" for s in valid_subjects]) + f" | {'Average':<11} |"
    sep_str = "| " + "-" * col_w_first + " | " + " | ".join(["-" * col_w for _ in valid_subjects]) + " | " + "-" * 11 + " |"
    
    print("\n\n" + "=" * len(header_str))
    print(f"{'📊 V2.0 学科细粒度性能矩阵 (Cognitive Subject-Level Matrix) 📊':^{len(header_str)}}")
    print("=" * len(header_str))
    print(header_str)
    print(sep_str)
    
    total_qs = sum(subject_metrics[s]["total"] for s in valid_subjects)
    qs_row = f"| {'Qs (Count)':<{col_w_first}} | " + " | ".join([f"{subject_metrics[s]['total']:<{col_w}}" for s in valid_subjects]) + f" | {total_qs:<11} |"
    print(qs_row)
    print(sep_str)
    
    # 构建各个模型的行并按 Average 从低到高排序
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
        # 👑 给全局最高胜率的威权大架构师戴上皇冠
        if j == engine.global_architect_idx: m_name = "👑 " + m_name 
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' or ord(c) > 0xFFFF else 1 for c in m_name)
        if display_width > col_w_first: 
            m_name = m_name[:col_w_first-2] + ".."
            display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' or ord(c) > 0xFFFF else 1 for c in m_name)
        
        padding = max(0, col_w_first - display_width)
        row_str = f"| {m_name}{' ' * padding} | " + " | ".join([f"{v:<{col_w}}" for v in row_vals]) + f" | {avg:.2f}%{' ':>4} |"
        print(row_str)
        
    print(sep_str)
    
    # 构建 V2.0 指标统计行
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
            
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in name)
        padding = max(0, col_w_first - display_width)
        row_str = f"| {name}{' ' * padding} | " + " | ".join([f"{v:<{col_w}}" for v in row_vals]) + f" | {avg_str:<11} |"
        print(row_str)
        
    print_bh_row("BH-V2-Exec (纯波形执行者)", "bh_exec")
    print_bh_row("BH-V2-Final (正交法庭融合)", "bh_final")
    print_bh_row("Gain (vs Best Pool Expert)", "gain", is_gain=True)
    print_bh_row("Early-Exit (波形早退数)", "early_exit", is_pct=False)
    print_bh_row("Ortho-Rescue (正交接管数)", "orthogonal_rescue", is_pct=False)
    print_bh_row("Fallback (威权兜底数)", "fallback_rescue", is_pct=False)
    print_bh_row("Total Rescue (扭转死局总数)", "rescue", is_pct=False)
    
    print("=" * len(header_str))

    if total_qs > 0:
        print("\n🏆 [Bench-Harness V2.0 拓扑路由与几何法庭验证大盘] 🏆")
        print("-" * 80)
        
        best_single_acc = max([m[0] for m in model_stats])
        bh_final_acc = sum([subject_metrics[s]["bh_final"] for s in valid_subjects]) / total_qs * 100
        global_early_exit = sum([subject_metrics[s]["early_exit"] for s in valid_subjects])
        global_orthogonal_rescue = sum([subject_metrics[s]["orthogonal_rescue"] for s in valid_subjects])
        global_fallback_rescue = sum([subject_metrics[s]["fallback_rescue"] for s in valid_subjects])

        print(f"  盲测集总题量      : {total_qs} 题")
        print(f"  🌟 V2 最终融合胜率 : {bh_final_acc:.2f}%")
        print("-" * 80)
        print(f"  🔒 [文本免污染] 触发 DTW 波形极速早退 : {global_early_exit} 次")
        print(f"  ⚖️ [跨维度打击] Gram 行列式正交推举法官 : {global_orthogonal_rescue} 次 (极其性感的代数降维打击！)")
        print(f"  🛡️ [防坍缩保护] 空间塌陷威权兜底接管   : {global_fallback_rescue} 次")
        print("-" * 80)

        diff = bh_final_acc - best_single_acc
        if diff > 0:
            print(f"  📈 相对池内【最强】单体模型天花板   : +{diff:.2f}% (完美逆转维度诅咒，重回 SOTA！)")
        else:
            print(f"  📉 距离池内【最强】单体模型天花板   : {diff:.2f}% (底座已免疫崩盘风险)")
            
        print("=" * 80)

if __name__ == "__main__":
    main()