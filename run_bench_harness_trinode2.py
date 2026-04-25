# -*- coding: utf-8 -*-
"""
Bench-Harness: 基于实例级正交共识的智能体编排底座
(🌟 终极版：深度创新二(早退自信锁) + 深度创新三(三节点上诉仲裁官 Tri-Node Escalation))
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
        """兼容性提取客观题答案"""
        if isinstance(ans, list):
            return "".join([str(a).strip().upper() for a in ans])
        return str(ans).strip().upper()

    @staticmethod
    def load_and_align(data_dir: str, models: list, subjects_filter: list = None):
        print(f"\n[*] 正在从 {data_dir} 加载并严格对齐多专家数据...")
        anchor_model = models[0]
        search_pattern = os.path.join(data_dir, f"{anchor_model}_*.json")
        anchor_files = glob.glob(search_pattern)
        
        datasets = {}
        total_aligned = 0
        missing_log = {}
        
        if not anchor_files:
            print(f"[!] 致命错误：在 {data_dir} 下没有找到基准模型 {anchor_model} 的任何 JSON 文件！请检查路径。")
            return datasets

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
            missing_models = []
            
            for model in models:
                model_file = os.path.join(data_dir, f"{model}_{keyword}.json")
                if not os.path.exists(model_file):
                    missing_models.append(model)
                    valid_dataset = False
                    continue
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            task_data[model] = data.get("example", data.get("examples", data))
                        else:
                            task_data[model] = data
                            
                        if not isinstance(task_data[model], list):
                            valid_dataset = False
                except Exception:
                    valid_dataset = False
                    
            if not valid_dataset: 
                for m in missing_models: missing_log[m] = missing_log.get(m, 0) + 1
                continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            
            min_len = min(len(task_data[m]) for m in models)
            if min_len == 0: continue
            
            for idx in range(min_len):
                item = base_examples[idx]
                q_text = item.get("question", "").strip()
                if not q_text: continue
                
                std_ans_raw = item.get("standard_answer", item.get("answer", []))
                std_ans_str = GaokaoDataLoader.extract_answer(std_ans_raw)
                
                row_corr = []
                row_mod_ans = []
                
                for model in models:
                    m_examples = task_data[model]
                    m_ans_raw = m_examples[idx].get("model_answer", m_examples[idx].get("answer", []))
                    m_ans_str = GaokaoDataLoader.extract_answer(m_ans_raw)
                    
                    is_correct = 1 if std_ans_str and std_ans_str == m_ans_str else 0
                    row_corr.append(is_correct)
                    row_mod_ans.append(m_ans_raw)
                    
                aligned_items.append({
                    "original_item": item,
                    "question": q_text,
                    "std_ans_raw": std_ans_raw,
                    "std_ans_str": std_ans_str,
                    "model_corr": row_corr,
                    "model_ans": row_mod_ans
                })
                    
            if aligned_items:
                with open(anchor_file, 'r', encoding='utf-8') as f:
                    p_data = json.load(f)
                    prompt = p_data.get("prompt", "") if isinstance(p_data, dict) else ""
                
                datasets[keyword] = {
                    "anchor_file_path": anchor_file,
                    "prompt": prompt,
                    "items": aligned_items
                }
                total_aligned += len(aligned_items)
                
        print(f"\n[+] 加载完毕。成功对齐 {total_aligned} 道题目，涉及完整合规学科 {len(datasets)} 个。")
        
        if missing_log:
            print("\n" + "!" * 80)
            print("[⚠️ 警告] 以下模型缺失部分学科文件，相关学科已自动跳过：")
            for m, count in sorted(missing_log.items(), key=lambda x: -x[1]): print(f"    - [{m}]: 缺失 {count} 个")
            print("!" * 80 + "\n")
            
        return datasets


# =====================================================================
# 2. 核心算法：Bench-Harness 引擎 (三节点仲裁官版)
# =====================================================================
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
        
        # 【计算宏观错误正交性先验】
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

        # 👑 【深度创新三：确立全局最高架构师】
        self.architect_idx = int(np.argmax(self.g_acc))
        print(f"\n👑 [系统分配] 经校准集宏观排位，全局准确率最高的神级基座 (Supreme Architect) 已自动锁定为: \n   >>> [{args.models[self.architect_idx]}] (全局基座胜率: {self.g_acc[self.architect_idx]*100:.2f}%) <<<")

        print(f"[*] 正在加载向量模型并提取情节记忆特征 (Model: {self.args.embed_model})...")
        self.embedder = SentenceTransformer(self.args.embed_model, device='cpu')
        calib_questions = [item["question"] for item in self.calib_items]
        self.V_calib = self.embedder.encode(calib_questions, show_progress_bar=True)

    def route(self, q_new_text: str):
        # 【在线微观流形坍缩检索】
        q_new_emb = self.embedder.encode([q_new_text], show_progress_bar=False)[0]
        sims = cosine_similarity(q_new_emb.reshape(1, -1), self.V_calib)[0]
        top_k_indices = np.argsort(sims)[-self.K:]
        
        Y_local = self.Y_calib[top_k_indices]
        local_scores = np.sum(Y_local, axis=0)
        exec_idx = int(np.argmax(local_scores))
        
        # 🌟 获取执行者局部胜任力 (用于触发早退机制)
        exec_local_acc = local_scores[exec_idx] / self.K
        
        # 【贝叶斯平滑与信心感知动态仲裁】
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


# =====================================================================
# 3. 主流程入口 (Main Routine)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Bench-Harness Tri-Node Escalation Simulation")
    
    parser.add_argument("--calib_dir", type=str, default="./GAOKAO-Bench-2023-2024/Data")
    parser.add_argument("--test_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--output_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    
    parser.add_argument("--top_k_ratio", type=float, default=0.30)
    parser.add_argument("--top_k_fixed", type=int, default=-1)
    parser.add_argument("--lambda_smooth", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--early_exit_threshold", type=float, default=0.85, help="早退锁阈值 (默认: 0.85)")
    
    args = parser.parse_args()

    print("=" * 140)
    print(f"{'🚀 Bench-Harness 状态机启动: [创新二] 流形熵早退锁 + [创新三] 三节点上诉仲裁官 🚀':^130}")
    print("=" * 140)
    print(f"| Experts ({len(args.models)}): {args.models}")
    print(f"| Params : top_k_ratio={args.top_k_ratio}, α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lambda_smooth}")
    print(f"| 🔒 Confidence Lock: {args.early_exit_threshold*100}%")
    if args.subjects:
        print(f"| Subjects Filter: {args.subjects}")
    print("-" * 140)

    calib_datasets = GaokaoDataLoader.load_and_align(args.calib_dir, args.models, args.subjects)
    if not calib_datasets: return
        
    engine = BenchHarnessEngine(calib_datasets, args)
    architect_idx = engine.architect_idx
    architect_model = args.models[architect_idx]
    
    test_datasets = GaokaoDataLoader.load_and_align(args.test_dir, args.models, args.subjects)
    if not test_datasets: return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_total_q = 0
    global_expert_corrects = np.zeros(len(args.models))
    global_bh_exec_corrects = 0
    global_bh_rev_corrects = 0   # 🌟 新增审查者全局准确率变量
    global_bh_consensus_corrects = 0
    
    global_rescue_counts = 0
    global_early_exit_counts = 0  # 🌟 早退次数
    global_escalation_counts = 0  # 🌟 上诉次数
    
    subject_metrics = {}
    print(f"\n🏃‍♂️ 正在执行 OOD 考题的在线微观路由与【三节点流转仲裁】仿真...")
    
    for keyword, data in test_datasets.items():
        items = data["items"]
        anchor_file_path = data["anchor_file_path"]
        bh_output_examples = []
        
        # 🌟 字典中新增了 bh_rev 键
        subject_metrics[keyword] = {
            "total": 0, "experts": np.zeros(len(args.models)),
            "bh_exec": 0, "bh_rev": 0, "bh_final": 0, 
            "rescue": 0, "early_exit": 0, "escalation": 0
        }
        
        short_desc = keyword[:30] + ".." if len(keyword) > 32 else keyword
        for item in tqdm(items, desc=f"Evaluating [{short_desc:<32}]"):
            global_total_q += 1
            subject_metrics[keyword]["total"] += 1
            
            q_text = item["question"]
            std_ans_str = item["std_ans_str"]
            
            for j in range(len(args.models)):
                corr = item["model_corr"][j]
                global_expert_corrects[j] += corr
                subject_metrics[keyword]["experts"][j] += corr
                
            # 1. 路由分配：获取执行者、审查者和流形胜率
            exec_idx, rev_idx, exec_local_acc = engine.route(q_text)
            
            exec_model = args.models[exec_idx]
            rev_model = args.models[rev_idx]
            
            exec_ans_raw = item["model_ans"][exec_idx]
            rev_ans_raw = item["model_ans"][rev_idx]
            arch_ans_raw = item["model_ans"][architect_idx] # 获取大架构师的历史作答底牌
            
            exec_corr = item["model_corr"][exec_idx]
            rev_corr = item["model_corr"][rev_idx]  # 🌟 提取当前审查者的对错状态
            
            global_bh_exec_corrects += exec_corr
            subject_metrics[keyword]["bh_exec"] += exec_corr
            
            global_bh_rev_corrects += rev_corr      # 🌟 累加审查者命中总数
            subject_metrics[keyword]["bh_rev"] += rev_corr
            
            exec_ans_str = GaokaoDataLoader.extract_answer(exec_ans_raw)
            rev_ans_str = GaokaoDataLoader.extract_answer(rev_ans_raw)
            arch_ans_str = GaokaoDataLoader.extract_answer(arch_ans_raw)
            
            action_log = ""
            is_early_exit = False
            is_escalated = False
            
            # ========================================================================
            # 🌟 【核心状态机流转：早退 -> 警报冲突 -> 架构师上诉】 🌟
            # ========================================================================
            if exec_local_acc >= args.early_exit_threshold:
                # 状态 1：低熵顺境。触发 [自信锁] 早退，屏蔽一切杂音！
                final_ans_raw = exec_ans_raw
                final_ans_str = exec_ans_str
                is_early_exit = True
                
                global_early_exit_counts += 1
                subject_metrics[keyword]["early_exit"] += 1
                action_log = f"【Yield Bypass (自信锁早退)】执行者局部胜率极高({exec_local_acc*100:.0f}%)，系统直接采纳初稿，越过审查阶段。"
            else:
                # 状态 2：高熵逆境，产生分歧 (Collision)！
                if exec_ans_str != rev_ans_str:
                    # 💥 冲突爆发！挂起任务，剥夺弱模型的话语权，提交大架构师 [Escalation]
                    is_escalated = True
                    final_ans_raw = arch_ans_raw
                    final_ans_str = arch_ans_str
                    
                    global_escalation_counts += 1
                    subject_metrics[keyword]["escalation"] += 1
                    action_log = f"【Architect Escalation (上诉仲裁)】前线执行者与审查者发生分歧！任务挂起，移交最高法官 [{architect_model}] 进行终局定夺。"
                else:
                    # 状态 3：高熵逆境，但未发生冲突 (Consensus)
                    final_ans_raw = exec_ans_raw
                    final_ans_str = exec_ans_str
                    action_log = f"【Consensus (达成共识)】执行者陷入高熵盲区，但红方审查者并未拉响分歧警报，维持原判。"
            # ========================================================================
                
            final_corr = 1 if std_ans_str and std_ans_str == final_ans_str else 0
            
            global_bh_consensus_corrects += final_corr
            subject_metrics[keyword]["bh_final"] += final_corr
            
            # 真实抢救次数：由于执行者错了，但由于触发了架构师上诉，由架构师给出了正确答案
            if exec_corr == 0 and final_corr == 1:
                global_rescue_counts += 1
                subject_metrics[keyword]["rescue"] += 1
                
            # 【持久化日志与 JSON 克隆】
            ex_out = item["original_item"].copy()
            ex_out["model_answer"] = final_ans_raw if isinstance(final_ans_raw, list) else [final_ans_str]
            ex_out["model_output"] = (
                f"【Bench-Harness 三节点流形状态机轨迹】\n"
                f"1. 局部流形评估: 执行者 [{exec_model}] 胜率 {exec_local_acc*100:.1f}%\n"
                f"2. 节点A (执行者) 草稿: {exec_ans_raw}\n"
                f"3. 节点B (审查者: {rev_model}) 意见: {rev_ans_raw if not is_early_exit else '（触发自信锁，未唤醒）'}\n"
                f"4. 节点C (架构师: {architect_model}) 裁决卷宗: {arch_ans_raw if is_escalated else '（未触发上诉，未唤醒）'}\n"
                f"5. 状态机流转日志: {action_log}\n"
                f"6. 终局输出答案: {final_ans_raw}"
            )
            bh_output_examples.append(ex_out)
            
        with open(anchor_file_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)
            
        save_data = p_data.copy()
        save_data["model_name"] = "Bench-Harness"
        
        if "example" in save_data: save_data["example"] = bh_output_examples
        elif "examples" in save_data: save_data["examples"] = bh_output_examples
        else: save_data["example"] = bh_output_examples
            
        save_path = os.path.join(args.output_dir, f"Bench-Harness_{keyword}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            
    # =====================================================================
    # 4. 横向超宽矩阵排版打印 (完美适配您的命令行输出格式)
    # =====================================================================
    sorted_subjects = sorted(subject_metrics.keys())
    valid_subjects = [subj for subj in sorted_subjects if subject_metrics[subj]["total"] > 0]
    
    col_w_first = 28
    col_w = 14
    header_str = f"| {'Model / Metric':<{col_w_first}} | " + " | ".join([f"{s.replace('2010-2022_', '').replace('2010-2013_', '').replace('2012-2022_', '')[:10]+'..':<{col_w}}" for s in valid_subjects]) + f" | {'Average':<11} |"
    sep_str = "| " + "-" * col_w_first + " | " + " | ".join(["-" * col_w for _ in valid_subjects]) + " | " + "-" * 11 + " |"
    
    print("\n\n" + "=" * len(header_str))
    print(f"{'📊 学科细粒度性能矩阵 (Subject-Level Performance Breakdown) 📊':^{len(header_str)}}")
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
    import unicodedata
    for avg, model, row_vals, j in model_stats:
        m_name = model
        # 👑 给大架构师戴上皇冠
        if j == engine.architect_idx: m_name = "👑 " + m_name 
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' or ord(c) > 0xFFFF else 1 for c in m_name)
        if display_width > col_w_first: 
            m_name = m_name[:col_w_first-2] + ".."
            display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' or ord(c) > 0xFFFF else 1 for c in m_name)
        
        padding = max(0, col_w_first - display_width)
        row_str = f"| {m_name}{' ' * padding} | " + " | ".join([f"{v:<{col_w}}" for v in row_vals]) + f" | {avg:.2f}%{' ':>4} |"
        print(row_str)
        
    print(sep_str)
    
    # 构建BH指标统计行
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
        
    print_bh_row("BH-Exec (纯执行者底座)", "bh_exec")
    # 🌟 在此插入审查者的行，精准放在两者的中间
    print_bh_row("BH-Rev (交叉审查者底座)", "bh_rev") 
    print_bh_row("BH-Final (三节点状态机定夺)", "bh_final")
    print_bh_row("Gain (vs Best Exp)", "gain", is_gain=True)
    print_bh_row("Early-Exit (自信锁早退数)", "early_exit", is_pct=False)
    print_bh_row("Escalation (上诉法庭接管数)", "escalation", is_pct=False)
    print_bh_row("Rescue (扭转死局总数)", "rescue", is_pct=False)
    
    print("=" * len(header_str))
    
    if total_qs > 0:
        print("\n🏆 [宏观 OOD 泛化大盘评估报告 (Tri-Node Generalization Report)] 🏆")
        print(f"盲测集总规模: {total_qs} 题")
        print("-" * 80)
        
        best_single_acc = max([m[0] for m in model_stats])
        bh_exec_acc = sum([subject_metrics[s]["bh_exec"] for s in valid_subjects]) / total_qs * 100
        bh_rev_acc = sum([subject_metrics[s]["bh_rev"] for s in valid_subjects]) / total_qs * 100     # 🌟 汇总全域审查者准确率
        bh_final_acc = sum([subject_metrics[s]["bh_final"] for s in valid_subjects]) / total_qs * 100
        global_escalation = sum([subject_metrics[s]["escalation"] for s in valid_subjects])
        global_early_exit = sum([subject_metrics[s]["early_exit"] for s in valid_subjects])
        
        print("【Bench-Harness 智能体编排底座 (三节点反脆弱状态机)】")
        print(f"  👑 最高架构师 (Supreme Architect)  : {architect_model}")
        print(f"  🌟 第一防线：仅执行者微观胜率       : {bh_exec_acc:.2f}%")
        print(f"  🛡️ 第二防线：仅审查者交叉胜率       : {bh_rev_acc:.2f}%")    # 🌟 增加在底部的全局展示
        print(f"  🚀 终极防线：三节点共识环最终成团   : {bh_final_acc:.2f}%")
        print("-" * 80)
        print(f"  🔒 [低熵顺境] 高置信度下强行早退越过审查次数   : {global_early_exit} 次 (完美切断了噪音污染的链路)")
        print(f"  ⚖️ [高熵分歧] 冲突爆发、上诉至架构师裁决次数   : {global_escalation} 次 (剥夺差生的一票否决权，化解过度纠错毒性)")
        print("-" * 80)
        
        diff = bh_final_acc - best_single_acc
        if diff > 0:
            print(f"  📈 相对池内【最强】单体模型天花板   : +{diff:.2f}% (完美逆转维度诅咒，重回 SOTA！)")
        else:
            print(f"  📉 距离池内【最强】单体模型天花板   : {diff:.2f}% (底座已免疫崩盘风险)")
            
        print("=" * 80)
        print(f"[*] ✅ 所有带有【三节点状态机诊断轨迹】的文件已保存至: {args.output_dir}/Bench-Harness_*.json")

if __name__ == "__main__":
    main()