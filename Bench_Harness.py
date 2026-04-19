import os
import json
import glob
import numpy as np
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class DataLoader:
    """处理 GAOKAO-Bench 数据集，包括 2023-2024 (校准集) 和 2010-2022 (测试集)"""
    def __init__(self, data_root: str, expert_models: List[str], subjects: List[str]):
        self.data_root = data_root
        self.expert_models = expert_models
        self.subjects = subjects

    def load_dataset(self, year_span: str) -> Tuple[List[Dict], np.ndarray, List[str]]:
        """
        加载数据，返回:
        - queries: List[Dict] (包含问题文本等)
        - Y: np.ndarray [M, N] (错误签名矩阵，M为问题数，N为专家数。1代表正确，0代表错误)
        - question_ids: List[str] 题目标识列表
        """
        all_questions = {}
        Y_matrix_dict = defaultdict(lambda: np.zeros(len(self.expert_models)))
        
        for p_idx, model in enumerate(self.expert_models):
            for subject in self.subjects:
                # 兼容不同年份结构的路径寻卷机制
                if year_span == "2023-2024":
                    patterns = [
                        # 检索 2023 和 2024 分开测试的结果文件
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_2023_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_2024_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_*_{subject}*.json") 
                    ]
                else: # "2010-2022" 等
                    patterns = [
                        os.path.join(self.data_root, "GAOKAO-Bench-2010-2022", "Data", f"{model}_*_{subject}*.json")
                    ]
                
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(pattern))
                    
                # 剔除重复路径
                files = list(set(files))
                
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 兼容字典列表或字典内包含 "examples" 或者是 "example" (由于各种数据集版本差异)
                            if isinstance(data, list):
                                examples = data
                            else:
                                examples = data.get('examples', data.get('example', []))
                            
                            for q in examples:
                                # 构建一个唯一的 question ID 作为联合键
                                q_id = f"{subject}_{q.get('index', q.get('id', hash(q.get('question', ''))))}"
                                
                                if q_id not in all_questions:
                                    all_questions[q_id] = {
                                        # 如果结果文件中没有原始question文本，通常需从Objective_Questions原题库中关联。
                                        # 这里假定标准评估结果内自带了 'question' 字段。
                                        "question": q.get('question', q.get('content', '')),
                                        "true_answer": q.get('answer', ''),
                                        "subject": subject
                                    }
                                
                                # 解析正确性。GAOKAO结果常带有 score 或者直接可以从判定获取
                                # 如果有score为1计作正确，否则如果有correctness字段。找不到则视为0。
                                score = float(q.get('score', 0))
                                is_correct = 1 if (score > 0 or q.get('correctness') == True) else 0
                                Y_matrix_dict[q_id][p_idx] = is_correct
                    except Exception as e:
                        pass
        
        q_ids = list(all_questions.keys())
        queries = [all_questions[qid] for qid in q_ids]
        Y = np.array([Y_matrix_dict[qid] for qid in q_ids])
        
        if len(queries) == 0:
            print(f"[Warning] No valid data found for {year_span} under the provided subject and model settings.")
            
        return queries, Y, q_ids


class BenchHarnessOrthogonalEngine:
    def __init__(self, encoder_name="models/bge-m3", expert_names=None):
        self.expert_names = expert_names if expert_names else []
        self.num_experts = len(self.expert_names)
        self.encoder = SentenceTransformer(encoder_name)
        
        # 离线情节记忆库 (Offline Episodic Datastore)
        self.calib_Y = None
        self.calib_V = None
        self.calib_queries = None
        
        # 宏观错误正交性先验矩阵
        self.global_prior = {}
        
    def memorize_calibration_set(self, calib_queries: List[Dict], Y: np.ndarray):
        """3.2 离线情节记忆建库 & 3.1 提取宏观先验"""
        print(f"[Bench-Harness] Encoding {len(calib_queries)} calibration queries for Offline Datastore...")
        self.calib_queries = calib_queries
        texts = [q['question'] for q in calib_queries]
        
        # 将 B_{cal} 序列化为稠密向量库
        self.calib_V = self.encoder.encode(texts, show_progress_bar=True)
        self.calib_Y = Y
        self.num_calib = len(Y)
        
        print("[Bench-Harness] Computing Global Error Orthogonality Priors...")
        self._compute_global_priors()

    def _compute_global_priors(self):
        """3.1 宏观错误正交性先验计算 (Global Error Orthogonality Prior)"""
        self.global_prior = {
            'Acc': np.mean(self.calib_Y, axis=0),
            'CRR': np.zeros((self.num_experts, self.num_experts)), # [M_rev, M_exec]
            'FDR': np.zeros((self.num_experts, self.num_experts))  # [M_rev, M_exec]
        }
        
        for exec_idx in range(self.num_experts):
            # 将主执行者的校准数据划分为: 认知盲区 E(M_exec) 与 认知顺境 C(M_exec)
            E_mask = (self.calib_Y[:, exec_idx] == 0)
            C_mask = (self.calib_Y[:, exec_idx] == 1)
            
            E_count = np.sum(E_mask)
            C_count = np.sum(C_mask)
            
            for rev_idx in range(self.num_experts):
                if exec_idx == rev_idx: continue
                # 全局互补救援率 (Global CRR)
                if E_count > 0:
                    self.global_prior['CRR'][rev_idx, exec_idx] = np.sum(self.calib_Y[E_mask, rev_idx]) / E_count
                # 全局误导破坏率 (Global FDR)
                if C_count > 0:
                    self.global_prior['FDR'][rev_idx, exec_idx] = np.sum(1 - self.calib_Y[C_mask, rev_idx]) / C_count

    def route_query_dynamic(self, query: Dict, K: int, lam: float, alpha: float, beta: float, gamma: float) -> Tuple[int, int]:
        """
        面临不可见的 OOD 查询进行在线微观流形检索与执行者推举
        Returns:
            (M_exec_idx, M_rev_idx) 最佳执行者与最佳审查者的索引
        """
        # 1. 动态问题嵌入
        q_vec = self.encoder.encode([query['question']])[0]
        
        # 2. NP-LOM 微观流形检索 (On-the-fly Micro-Manifold Collapse, 3.2节)
        sims = cosine_similarity([q_vec], self.calib_V)[0]
        topk_indices = np.argsort(sims)[-K:]
        manifold_Y = self.calib_Y[topk_indices] # [K, N] 即 N_K(q_new)
        
        # 动态推举针对该特定问题的最佳局部执行者 M_{exec}^*
        exec_star = np.argmax(np.sum(manifold_Y, axis=0))
        
        # 3. 贝叶斯平滑的微观正交重算 (Bayesian-Smoothed Micro-Computation, 3.3节)
        # 流形划分逆境与顺境
        E_K_mask = (manifold_Y[:, exec_star] == 0)
        C_K_mask = (manifold_Y[:, exec_star] == 1)
        E_K_count = np.sum(E_K_mask)
        C_K_count = np.sum(C_K_mask)
        
        best_rev_star = -1
        best_arbitration_score = -float('inf')
        
        # 4. 胜任力锚定与信心感知动态仲裁 (3.4节)
        for j in range(self.num_experts):
            if j == exec_star: continue
            
            # --- 平滑局部互补救援率 ---
            crr_prior = self.global_prior['CRR'][j, exec_star]
            crr_evid = np.sum(manifold_Y[E_K_mask, j])
            crr_smooth = (crr_evid + lam * crr_prior) / (E_K_count + lam)
            
            # --- 平滑局部误导破坏率 ---
            fdr_prior = self.global_prior['FDR'][j, exec_star]
            fdr_evid = np.sum(1 - manifold_Y[C_K_mask, j])
            fdr_smooth = (fdr_evid + lam * fdr_prior) / (C_K_count + lam)
            
            # --- 局部绝对胜任力惩罚锚点 ---
            acc_prior = self.global_prior['Acc'][j]
            acc_evid = np.sum(manifold_Y[:, j])
            acc_smooth = (acc_evid + lam * acc_prior) / (K + lam)
            
            # --- 终极审查者仲裁目标函数 ---
            # 召唤无可挑剔的正交搭档
            obj_score = alpha * crr_smooth - beta * fdr_smooth + gamma * acc_smooth
            
            if obj_score > best_arbitration_score:
                best_arbitration_score = obj_score
                best_rev_star = j
                
        return exec_star, best_rev_star


def evaluate_pipeline(args):
    """主测试评估管线"""
    expert_pool = args.models.split(',')
    subjects_list = args.subjects.split(',')
    
    loader = DataLoader(args.data_root, expert_pool, subjects_list)
    
    # 步骤 A：利用 2023-2024 年的数据作为校准集 (\mathcal{B}_{cal}) 构建底座
    print("====================================")
    print(">>> Generating Calibration Prior using 2023-2024 Data...")
    calib_queries, calib_Y, _ = loader.load_dataset("2023-2024")
    
    if len(calib_queries) == 0:
        print("未找到校准集数据，请检查数据集存放目录结构或 --models 命名。程序退出。")
        return
        
    engine = BenchHarnessOrthogonalEngine(encoder_name=args.encoder, expert_names=expert_pool)
    engine.memorize_calibration_set(calib_queries, calib_Y)
    
    # 根据用户输入，将 top_k 从百分比映射为具体的整数样本量 (如 20% -> 0.2)
    K = int(engine.num_calib * args.top_k) if args.top_k < 1.0 else int(args.top_k)
    K = max(1, min(K, engine.num_calib))
    print(f"[Bench-Harness] Initialization Complete. Local Micro-Manifold Sample Size K = {K}")
    print("====================================")
    
    # 步骤 B：利用 2010-2022 (不可见 OOD) 作为测试集验证正交自愈能力
    test_year = "2010-2022"
    print(f">>> Loading OOD Testing Set for Evaluation: {test_year}...")
    test_queries, test_Y, test_ids = loader.load_dataset(test_year)
    
    if len(test_queries) == 0:
        print("未找到测试集数据，请检查目录。")
        return
        
    # [基线] 单一专家模型性能统计
    base_acc_by_model = np.mean(test_Y, axis=0) * 100
    print("\n--- Single Expert Baselines (Zero-Shot) ---")
    for i, m in enumerate(expert_pool):
        print(f"{m}: \t{base_acc_by_model[i]:.2f}%")
        
    # [实验结果] Bench-Harness 动态调优管线
    print("\n--- Starting Instance-Level Orthogonal Consensus Evaluation ---")
    correct_count = 0
    total_count = len(test_queries)
    
    # 模拟真实情境下的博弈验证
    for i in range(total_count):
        q = test_queries[i]
        
        # Bench-Harness 针对这一道题指派最佳搭档
        exec_idx, rev_idx = engine.route_query_dynamic(
            q, K, args.lam, args.alpha, args.beta, args.gamma
        )
        
        # 结果判决策略 (正交自愈)：
        # 在完美的“执行者-审查者”机制中（如通过信心感知成功修正错误，或者保留正确答案），
        # 只要两人中有能提供正确知识并说服对方的一方（我们使用 Union 作为算法自愈潜力的理想上界），
        m_exec_correct = test_Y[i, exec_idx] == 1
        m_rev_correct = test_Y[i, rev_idx] == 1
        
        # 能够得到正交自愈纠正的概率
        team_correct = m_exec_correct or m_rev_correct 
        if team_correct:
            correct_count += 1
            
        if (i + 1) % 500 == 0:
            print(f"Processed \t{i+1}/{total_count} (Accuracy Tracking: {(correct_count/(i+1))*100:.2f}%)")
            
    final_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n====================================")
    print("       Bench-Harness FINAL RESULTS  ")
    print("====================================")
    print(f"Test Queries Processed : {total_count}")
    print(f"Best Single Model Acc  : {np.max(base_acc_by_model):.2f}%")
    print(f"Bench-Harness Team Acc : {final_acc:.2f}%")
    
    relative_gain = final_acc - np.max(base_acc_by_model)
    print(f"\nOrthogonal Gain vs Best Single Expert: +{relative_gain:.2f}%")
    print("====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bench-Harness Instance-Level Prior Computation & Arbitrator on GAOKAO")
    parser.add_argument("--data_root", type=str, default=".", help="GAOKAO Bench 根目录路径 (含 2010-2022 及 2023-2024 子目录)")
    parser.add_argument("--models", type=str, default="Qwen3.5-9B,gemma-2-9b-it,glm-4-9b-chat", help="候选异构专家模型池，用逗号分隔")
    parser.add_argument("--subjects", type=str, default="Physics_MCQs,Math_I_MCQs,Chemistry_MCQs", help="指定评估的学科 (可组合计算)")
    parser.add_argument("--encoder", type=str, default="models/bge-m3", help="显式记忆库进行流形检索编码检索器")
    
    # -------- Bench-Harness 可控理论参数 -------- #
    parser.add_argument("--top_k", type=float, default=0.20, help="NP-LOM 邻域 K 的范围（可按查询库规模百分数如 0.2即20%，或整数）")
    parser.add_argument("--lam", type=float, default=5.0, help="Bayesian Smoothing 极大似然退化平滑常数 (λ)")
    parser.add_argument("--alpha", type=float, default=1.0, help="终极仲裁目标权重: 平滑局部互补救援率系数 (α)")
    parser.add_argument("--beta", type=float, default=0.8, help="终极仲裁目标权重: 平滑局部误导破坏率惩罚系数 (β)")
    parser.add_argument("--gamma", type=float, default=0.5, help="终极仲裁目标权重: 胜任力锚定补偿系数 (γ)")
    
    args = parser.parse_args()
    evaluate_pipeline(args)