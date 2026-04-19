# -*- coding: utf-8 -*-
"""
Bench-Harness: 基于实例级正交共识的智能体编排底座
(🌟 深度创新二版：引入流形熵驱动的自信锁与早退机制 Entropy-Driven Confidence Lock & Early-Exit)
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
            # 排除分离的分片文件、主观题文件以及生成的 Bench-Harness 结果
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
                            print(f"  [格式警告] {model_file} 中没有找到有效的题目列表。")
                            valid_dataset = False
                except Exception as e:
                    print(f"  [读取报错] 读取 {model_file} 失败: {e}")
                    valid_dataset = False
                    
            if not valid_dataset: 
                print(f"[!] 跳过学科 [{keyword}] ---> 以下模型缺失文件或损坏: {missing_models}")
                for m in missing_models:
                    missing_log[m] = missing_log.get(m, 0) + 1
                continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            
            min_len = min(len(task_data[m]) for m in models)
            if min_len == 0:
                print(f"[!] 跳过学科 [{keyword}] ---> 至少有一个模型的题目数量为 0。")
                continue
            if min_len < len(base_examples):
                print(f"[*] 警告 [{keyword}] ---> 模型间答题数量不一致，已安全截断至最小公共长度: {min_len} 题。")
            
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
            print("[⚠️ 严重警告] 由于部分模型缺失测试 JSON 文件，导致多个学科被强行跳过！")
            print(">> 缺失文件统计清单：")
            for m, count in sorted(missing_log.items(), key=lambda x: -x[1]):
                print(f"    - 模型 [{m}]: 缺失 {count} 个学科文件")
            print(">> 解决方案：请检查上述模型，并从你的 --models 参数列表中移除它们，才能获得评估结果！")
            print("!" * 80 + "\n")
            
        return datasets


# =====================================================================
# 2. 核心算法：Bench-Harness 引擎
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
            raise ValueError("[❌ 失败] 校准集为空。请先排查上方的 ⚠️警告，在 --models 中剔除掉那些缺失文件的模型！")
            
        self.Y_calib = np.array([item["model_corr"] for item in self.calib_items])
        
        if self.args.top_k_fixed > 0:
            self.K = min(self.args.top_k_fixed, self.M)
        else:
            self.K = max(5, int(self.M * self.args.top_k_ratio))
        
        # 【3.1 宏观错误正交性先验】
        self.g_acc = np.mean(self.Y_calib, axis=0)
        self.g_crr = np.zeros((self.N, self.N))
        self.g_fdr = np.zeros((self.N, self.N))
        
        for exec_idx in range(self.N):
            err_mask = (self.Y_calib[:, exec_idx] == 0)
            corr_mask = (self.Y_calib[:, exec_idx] == 1)
            num_err = np.sum(err_mask)
            num_corr = np.sum(corr_mask)
            
            for rev_idx in range(self.N):
                if exec_idx == rev_idx: continue
                if num_err > 0:
                    self.g_crr[exec_idx, rev_idx] = np.sum(self.Y_calib[err_mask, rev_idx] == 1) / num_err
                if num_corr > 0:
                    self.g_fdr[exec_idx, rev_idx] = np.sum(self.Y_calib[corr_mask, rev_idx] == 0) / num_corr

        print(f"[*] 正在加载本地向量模型并提取情节记忆特征 (Model: {self.args.embed_model})...")
        self.embedder = SentenceTransformer(self.args.embed_model)
        calib_questions = [item["question"] for item in self.calib_items]
        self.V_calib = self.embedder.encode(calib_questions, show_progress_bar=True)

    def route(self, q_new_text: str):
        # 【3.2 在线微观流形坍缩检索】
        q_new_emb = self.embedder.encode([q_new_text], show_progress_bar=False)[0]
        sims = cosine_similarity(q_new_emb.reshape(1, -1), self.V_calib)[0]
        top_k_indices = np.argsort(sims)[-self.K:]
        
        Y_local = self.Y_calib[top_k_indices]
        local_scores = np.sum(Y_local, axis=0)
        exec_idx = int(np.argmax(local_scores))
        
        # 🌟 获取局部胜任力 (用于触发置信度早退机制)
        exec_local_acc = local_scores[exec_idx] / self.K
        
        # 【3.3 & 3.4 贝叶斯平滑与信心感知动态仲裁】
        err_mask = (Y_local[:, exec_idx] == 0)
        corr_mask = (Y_local[:, exec_idx] == 1)
        num_err = np.sum(err_mask)
        num_corr = np.sum(corr_mask)
        
        best_rev_idx = -1
        best_obj_score = -float('inf')
        lam = self.args.lambda_smooth
        eps = 1e-9 
        
        for j in range(self.N):
            if j == exec_idx: continue
            
            local_rescues = np.sum(Y_local[err_mask, j] == 1)
            crr_hat = (local_rescues + lam * self.g_crr[exec_idx, j]) / (num_err + lam + eps)
            
            local_disrupts = np.sum(Y_local[corr_mask, j] == 0)
            fdr_hat = (local_disrupts + lam * self.g_fdr[exec_idx, j]) / (num_corr + lam + eps)
            
            local_corrects = np.sum(Y_local[:, j] == 1)
            acc_hat = (local_corrects + lam * self.g_acc[j]) / (self.K + lam + eps)
            
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
    parser = argparse.ArgumentParser(description="Bench-Harness Confidence Lock & Early-Exit Simulation")
    
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
    
    # 🌟 创新参数：自信锁阈值
    parser.add_argument("--early_exit_threshold", type=float, default=0.85, help="自信锁阈值(胜率>=该值触发早退跳过审查) (默认: 0.85)")
    
    args = parser.parse_args()

    print("=" * 120)
    print(f"{'🚀 Bench-Harness 实例级正交共识 盲测验证启动 (引入流形熵早退机制)':^110}")
    print("=" * 120)
    print(f"| Experts ({len(args.models)}): {args.models}")
    print(f"| Params : top_k_ratio={args.top_k_ratio}, α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lambda_smooth}")
    print(f"| 🔒 Confidence Lock Threshold: {args.early_exit_threshold*100}%")
    if args.subjects:
        print(f"| Subjects Filter: {args.subjects}")
    print("-" * 120)

    calib_datasets = GaokaoDataLoader.load_and_align(args.calib_dir, args.models, args.subjects)
    if not calib_datasets: return
        
    engine = BenchHarnessEngine(calib_datasets, args)
    
    test_datasets = GaokaoDataLoader.load_and_align(args.test_dir, args.models, args.subjects)
    if not test_datasets: return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_total_q = 0
    global_expert_corrects = np.zeros(len(args.models))
    global_bh_exec_corrects = 0
    global_bh_consensus_corrects = 0
    global_rescue_counts = 0
    global_early_exit_counts = 0  # 🌟 记录早退次数
    
    subject_metrics = {}
    
    print(f"\n🏃‍♂️ 正在执行 OOD 考题的在线微观路由与共识纠错仿真...")
    
    for keyword, data in test_datasets.items():
        items = data["items"]
        anchor_file_path = data["anchor_file_path"]
        bh_output_examples = []
        
        subject_metrics[keyword] = {
            "total": 0, "experts": np.zeros(len(args.models)),
            "bh_exec": 0, "bh_final": 0, "rescue": 0, "early_exit": 0
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
                
            # 接收路由返回的结果及局部胜率
            exec_idx, rev_idx, exec_local_acc = engine.route(q_text)
            
            exec_model = args.models[exec_idx]
            rev_model = args.models[rev_idx]
            
            exec_ans_raw = item["model_ans"][exec_idx]
            rev_ans_raw = item["model_ans"][rev_idx]
            exec_corr = item["model_corr"][exec_idx]
            
            global_bh_exec_corrects += exec_corr
            subject_metrics[keyword]["bh_exec"] += exec_corr
            
            exec_ans_str = GaokaoDataLoader.extract_answer(exec_ans_raw)
            rev_ans_str = GaokaoDataLoader.extract_answer(rev_ans_raw)
            
            action_log = ""
            is_early_exit = False
            
            # ========================================================================
            # 🌟 【深度创新二：流形熵驱动的自信锁与早退机制 (Early-Exit)】 🌟
            # ========================================================================
            if exec_local_acc >= args.early_exit_threshold:
                # 胜率极高，说明在低熵顺境！触发自信锁 [YIELD Bypass]
                final_ans_raw = exec_ans_raw
                final_ans_str = exec_ans_str
                is_early_exit = True
                
                global_early_exit_counts += 1
                subject_metrics[keyword]["early_exit"] += 1
                action_log = f"【Early Exit (自信锁生效)】执行者在相似历史错题中胜率极高({exec_local_acc*100:.0f}%)，触发安全锁剥夺审查者干预权，直接采纳执行者答案。"
            else:
                # 若胜率不高（高熵），则挂起任务，允许审查者发言进行正交裁决
                if exec_ans_str != rev_ans_str:
                    final_ans_raw = rev_ans_raw
                    final_ans_str = rev_ans_str
                    action_log = f"【Override (纠错抢救)】执行者陷入高熵盲区，审查者 [{rev_model}] 强势接入并覆写了答案。"
                else:
                    final_ans_raw = exec_ans_raw
                    final_ans_str = exec_ans_str
                    action_log = f"【Consensus (达成共识)】执行者陷入高熵盲区，但审查者与其意见一致，维持原判。"
            # ========================================================================
                
            final_corr = 1 if std_ans_str and std_ans_str == final_ans_str else 0
            
            global_bh_consensus_corrects += final_corr
            subject_metrics[keyword]["bh_final"] += final_corr
            
            # 早退由于直接输出，不存在被审查者修改的情况，因此只有未早退时才会算 rescue
            if not is_early_exit and exec_corr == 0 and final_corr == 1:
                global_rescue_counts += 1
                subject_metrics[keyword]["rescue"] += 1
                
            # 【包装原格式】确保模型回答格式为原本的列表或字符串
            ex_out = item["original_item"].copy()
            ex_out["model_answer"] = final_ans_raw if isinstance(final_ans_raw, list) else [final_ans_str]
            ex_out["model_output"] = (
                f"【Bench-Harness 离线仿真轨迹 (流形熵感知架构)】\n"
                f"1. 局部流形评估: 执行者 [{exec_model}] 胜率 {exec_local_acc*100:.1f}%\n"
                f"2. 主执行者草稿: {exec_ans_raw}\n"
                f"3. 正交审查者意见: {rev_ans_raw if not is_early_exit else '（已触发自信锁，未唤醒审查者）'}\n"
                f"4. 仲裁动作: {action_log}\n"
                f"5. 仲裁系统输出答案: {final_ans_raw}"
            )
            bh_output_examples.append(ex_out)
            
        # 🌟 完美克隆 JSON 包装壳
        with open(anchor_file_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)
            
        save_data = p_data.copy()
        save_data["model_name"] = "Bench-Harness"
        
        # 智能匹配不同年份数据中的键名
        if "example" in save_data:
            save_data["example"] = bh_output_examples
        elif "examples" in save_data:
            save_data["examples"] = bh_output_examples
        else:
            save_data["example"] = bh_output_examples
            
        save_path = os.path.join(args.output_dir, f"Bench-Harness_{keyword}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            
    # =====================================================================
    # 4. 打印报告矩阵 
    # =====================================================================
    sorted_subjects = sorted(subject_metrics.keys())
    valid_subjects = [subj for subj in sorted_subjects if subject_metrics[subj]["total"] > 0]
    
    # 动态计算表格宽度
    table_width = 30 + 15 * len(valid_subjects) + 12
    
    print("\n\n" + "=" * table_width)
    print(f"{'📊 学科细粒度性能矩阵 (Subject-Level Performance Breakdown) 📊':^{table_width}}")
    print("=" * table_width)
    
    header = f"| {'Model / Metric':<26} |"
    for subj in valid_subjects:
        short_kw = subj.replace("2010-2022_", "").replace("2010-2013_", "").replace("2012-2022_", "")
        short_kw = short_kw[:10] + ".." if len(short_kw) > 12 else short_kw
        header += f" {short_kw:<12} |"
    header += f" {'Average':<9} |"
    print(header)
    
    sep_line = "|" + "-"*28 + "|"
    for _ in valid_subjects:
        sep_line += "-"*14 + "|"
    sep_line += "-"*11 + "|"
    print(sep_line)
    
    # 打印题目数量行
    qs_row = f"| {'Qs (Count)':<26} |"
    total_qs = 0
    for subj in valid_subjects:
        q_count = subject_metrics[subj]["total"]
        total_qs += q_count
        qs_row += f" {q_count:>12} |"
    qs_row += f" {total_qs:>9} |"
    print(qs_row)
    print(sep_line)
    
    # 构建各个模型的行并按 Average 从低到高排序
    model_rows = []
    for j, model in enumerate(args.models):
        short_name = model[:24] + ".." if len(model) > 26 else model
        row_str = f"| {short_name:<26} |"
        weighted_sum = 0.0
        
        for subj in valid_subjects:
            q_count = subject_metrics[subj]["total"]
            acc = (subject_metrics[subj]["experts"][j] / q_count) * 100
            weighted_sum += acc * q_count
            row_str += f" {acc:>11.2f}% |"
            
        weighted_avg = (weighted_sum / total_qs) if total_qs else 0.0
        row_str += f" {weighted_avg:>8.2f}% |"
        model_rows.append((weighted_avg, row_str))
        
    # 按 Average 升序排列
    model_rows.sort(key=lambda x: x[0])
    for avg, row_str in model_rows:
        print(row_str)
        
    print(sep_line)
    
    # 构建BH指标统计行
    def print_metric_row(name, key):
        import unicodedata
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in name)
        padding = max(0, 26 - display_width)
        row_str = f"| {name}{' ' * padding} |"
        weighted_sum = 0.0
        total_val = 0
        for subj in valid_subjects:
            q_count = subject_metrics[subj]["total"]
            if key in ["rescue", "early_exit"]:
                val = subject_metrics[subj][key]
                total_val += val
                row_str += f" {val:>12} |"
            elif key == "gain":
                bh_f = (subject_metrics[subj]["bh_final"] / q_count) * 100
                best_ex = max([(subject_metrics[subj]["experts"][idx] / q_count) * 100 for idx in range(len(args.models))])
                gain = bh_f - best_ex
                weighted_sum += gain * q_count
                gain_str = f"+{gain:.2f}%" if gain > 0 else f"{gain:.2f}%"
                row_str += f" {gain_str:>12} |"
            else:
                val = (subject_metrics[subj][key] / q_count) * 100
                weighted_sum += val * q_count
                row_str += f" {val:>11.2f}% |"
                
        if key in ["rescue", "early_exit"]:
            row_str += f" {total_val:>9} |"
        else:
            weighted_avg = (weighted_sum / total_qs) if total_qs else 0.0
            if key == "gain":
                gain_str = f"+{weighted_avg:.2f}%" if weighted_avg > 0 else f"{weighted_avg:.2f}%"
                row_str += f" {gain_str:>8} |"
            else:
                row_str += f" {weighted_avg:>8.2f}% |"
        print(row_str)
        
    print_metric_row("BH-Exec (执行者单独准确率)", "bh_exec")
    print_metric_row("BH-Final (加入审查与门控)", "bh_final")
    print_metric_row("Gain (vs Best Expert)", "gain")
    print_metric_row("Early-Exit Counts (早退数)", "early_exit")
    print_metric_row("Rescue Counts (有效救场数)", "rescue")
    
    print("=" * table_width)
    
    if global_total_q > 0:
        print("\n🏆 [宏观 OOD 泛化大盘评估报告 (Global OOD Generalization Report)] 🏆")
        print(f"盲测集总规模: {global_total_q} 题 (数据源自: {args.test_dir})")
        print("-" * 75)
        
        best_single_acc = np.max(global_expert_corrects) / global_total_q * 100
        bh_exec_acc = global_bh_exec_corrects / global_total_q * 100
        bh_final_acc = global_bh_consensus_corrects / global_total_q * 100
        exit_rate = global_early_exit_counts / global_total_q * 100
        
        print("【Bench-Harness 智能体正交编排底座 (引入自信锁机制)】")
        print(f"  🌟 仅执行者微观胜率 (原生底座实力) : {bh_exec_acc:.2f}%")
        print(f"  🚀 实例级正交共识环 (最终成团)     : {bh_final_acc:.2f}%")
        print(f"  🔥 正交审查者成功抢救逆境次数      : {global_rescue_counts} 次")
        print(f"  🔒 高置信度下强行早退越过审查次数  : {global_early_exit_counts} 次 (大幅降低被瞎猜带偏的负增益概率)")
        
        diff = bh_final_acc - best_single_acc
        if diff > 0:
            print(f"  📈 相对池内【最强】单体模型天花板   : +{diff:.2f}% (越级击杀！)")
        else:
            print(f"  📉 相对池内【最强】单体模型天花板   : {diff:.2f}% (可调节 --confidence_threshold 阈值)")
            
        print("=" * 75)
        print(f"[*] ✅ 所有带有【Early-Exit 轨迹诊断】的格式化文件已完美保存至:\n    {args.output_dir}/Bench-Harness_*.json")

if __name__ == "__main__":
    main()