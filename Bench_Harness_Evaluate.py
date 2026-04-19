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
    """处理GAOKAO-Bench数据格式的读取与预处理"""
    def __init__(self, data_root: str, expert_models: List[str], subjects: List[str]):
        self.data_root = data_root
        self.expert_models = expert_models
        self.subjects = subjects

    def load_dataset(self, year_span="2023-2024") -> Tuple[List[Dict], np.ndarray, List[str]]:
        """
        加载数据，返回:
        - queries: List[Dict] (包含问题文本、索引等)
        - Y: np.ndarray [M, N] (错误签名矩阵，M为问题数，N为专家数)
        - question_ids: List[str] 独一无二的题目标识
        """
        all_questions = {}
        Y_matrix_dict = defaultdict(lambda: np.zeros(len(self.expert_models)))
        
        for p_idx, model in enumerate(self.expert_models):
            for subject in self.subjects:
                if year_span == "2023-2024":
                    # 兼容 2023 和 2024 独立放在 GAOKAO-Bench-2010-2022/Data/ 下面的特殊目录结构
                    patterns = [
                        os.path.join(self.data_root, "GAOKAO-Bench-2010-2022", "Data", "GAOKAO-Bench-2023", f"{model}_*_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2010-2022", "Data", "GAOKAO-Bench-2024", f"{model}_*_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_*_{subject}*.json")
                    ]
                else:
                    patterns = [
                        os.path.join(self.data_root, f"GAOKAO-Bench-{year_span}", "Data", f"{model}_*_{subject}*.json")
                    ]
                
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(pattern))
                    
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 假设原始文件包含"example"列表或字典
                            examples = data if isinstance(data, list) else data.get('examples', [])
                            for q in examples:
                                # 构建一个唯一的question ID
                                q_id = f"{subject}_{q.get('index', q.get('id', hash(q.get('question', ''))))}"
                                
                                if q_id not in all_questions:
                                    all_questions[q_id] = {
                                        "question": q.get('question', ''),
                                        "true_answer": q.get('answer', ''),
                                        "subject": subject
                                    }
                                
                                # 判断是否正确 (这里需要根据你数据集的真实格式调整: 评估分数或文本匹配)
                                # 假设有 'score' 字段，1为对，0为错
                                score = float(q.get('score', 0))
                                Y_matrix_dict[q_id][p_idx] = 1 if score > 0 else 0
                    except Exception as e:
                        pass # 简单跳过解析错误的文件
        
        q_ids = list(all_questions.keys())
        queries = [all_questions[qid] for qid in q_ids]
        Y = np.array([Y_matrix_dict[qid] for qid in q_ids])
        
        # 兜底：如果没有读到抛出警告
        if len(queries) == 0:
            print(f"[Warning] No valid data found for {year_span}. Check file paths or model names.")
            
        return queries, Y, q_ids

class BenchHarnessOrthogonalEngine:
    def __init__(self, encoder_name="BAAI/bge-m3", expert_names=None):
        self.expert_names = expert_names if expert_names else []
        self.num_experts = len(self.expert_names)
        self.encoder = SentenceTransformer(encoder_name)
        
        # 离线记忆库
        self.calib_Y = None
        self.calib_V = None
        
        # 宏观先验字典
        self.global_prior = {}
        
    def memorize_calibration_set(self, calib_queries: List[Dict], Y: np.ndarray):
        """3.2/3.1 离线情节记忆建库 & 提取宏观先验"""
        print("[1/4] Encoding calibration set...")
        texts = [q['question'] for q in calib_queries]
        self.calib_V = self.encoder.encode(texts, show_progress_bar=True)
        self.calib_Y = Y
        self.num_calib = len(Y)
        
        print("[2/4] Computing Global Error Orthogonality Prior...")
        self._compute_global_priors()

    def _compute_global_priors(self):
        """3.1 宏观错误正交性先验计算"""
        self.global_prior = {
            'Acc': np.mean(self.calib_Y, axis=0),
            'CRR': np.zeros((self.num_experts, self.num_experts)), # [rev, exec] -> CRR(rev|exec)
            'FDR': np.zeros((self.num_experts, self.num_experts))  # [rev, exec] -> FDR(rev|exec)
        }
        
        for exec_i in range(self.num_experts):
            # 认知顺境与盲区
            blind_spot_mask = (self.calib_Y[:, exec_i] == 0)
            comfort_zone_mask = (self.calib_Y[:, exec_i] == 1)
            
            E_exec_count = np.sum(blind_spot_mask)
            C_exec_count = np.sum(comfort_zone_mask)
            
            for rev_j in range(self.num_experts):
                if exec_i == rev_j: continue
                # 全局互补救援率 (Global CRR)
                if E_exec_count > 0:
                    self.global_prior['CRR'][rev_j, exec_i] = np.sum(self.calib_Y[blind_spot_mask, rev_j]) / E_exec_count
                
                # 全局误导破坏率 (Global FDR)
                if C_exec_count > 0:
                    self.global_prior['FDR'][rev_j, exec_i] = np.sum(1 - self.calib_Y[comfort_zone_mask, rev_j]) / C_exec_count

    def route_query(self, query: Dict, K: int, lam: float, alpha: float, beta: float, gamma: float) -> Tuple[int, int]:
        """
        在线微观流形推举、重算与仲裁
        Returns: (M_exec_idx, M_rev_idx)
        """
        # 1. 编码查询
        q_vec = self.encoder.encode([query['question']])[0]
        
        # 2. NP-LOM 微观流形检索 (3.2)
        sims = cosine_similarity([q_vec], self.calib_V)[0]
        topk_indices = np.argsort(sims)[-K:]
        
        manifold_Y = self.calib_Y[topk_indices] # Shape: [K, N]
        
        # 动态推举特定问题最佳执行者 M_{exec}^*
        exec_star = np.argmax(np.sum(manifold_Y, axis=0))
        
        # 3. 贝叶斯平滑的微观正交重算 (3.3)
        E_K_mask = (manifold_Y[:, exec_star] == 0)
        C_K_mask = (manifold_Y[:, exec_star] == 1)
        
        E_K_count = np.sum(E_K_mask)
        C_K_count = np.sum(C_K_mask)
        
        best_rev = -1
        best_score = -float('inf')
        
        # 4. 胜任力锚定与信心感知动态仲裁 (3.4)
        for j in range(self.num_experts):
            if j == exec_star: continue
            
            # Smooth CRR
            crr_global = self.global_prior['CRR'][j, exec_star]
            crr_local_smooth = (np.sum(manifold_Y[E_K_mask, j]) + lam * crr_global) / (E_K_count + lam)
            
            # Smooth FDR
            fdr_global = self.global_prior['FDR'][j, exec_star]
            fdr_local_smooth = (np.sum(1 - manifold_Y[C_K_mask, j]) + lam * fdr_global) / (C_K_count + lam)
            
            # Smooth Accuracy (Competence Anchor)
            acc_global = self.global_prior['Acc'][j]
            acc_local_smooth = (np.sum(manifold_Y[:, j]) + lam * acc_global) / (K + lam)
            
            # 终极审查者仲裁函数
            obj_score = alpha * crr_local_smooth - beta * fdr_local_smooth + gamma * acc_local_smooth
            
            if obj_score > best_score:
                best_score = obj_score
                best_rev = j
                
        return exec_star, best_rev

def evaluate_pipeline(args):
    expert_pool = args.models.split(',')
    
    loader = DataLoader(args.data_root, expert_pool, args.subjects.split(','))
    
    # 构建校准集 (2023-2024)
    print(">>> Loading Calibration Set...")
    calib_queries, calib_Y, _ = loader.load_dataset("2023-2024")
    
    if len(calib_queries) == 0:
        return
        
    engine = BenchHarnessOrthogonalEngine(encoder_name=args.encoder, expert_names=expert_pool)
    engine.memorize_calibration_set(calib_queries, calib_Y)
    
    # 选择Top-K
    # 如果 args.top_k < 1，说明是使用千分比/百分比
    K = int(engine.num_calib * args.top_k) if args.top_k < 1.0 else int(args.top_k)
    K = max(1, min(K, engine.num_calib))
    print(f"[3/4] Setup complete. Local Manifold K = {K}")
    print("====================================")
    
    # 验证测试集 (OOD - 2010-2022)
    print(f">>> Loading Testing Set ({args.test_year})...")
    test_queries, test_Y, test_ids = loader.load_dataset(args.test_year)
    
    # 验证指标基线
    base_acc_by_model = np.mean(test_Y, axis=0) * 100
    print("--- Single Expert Baselines ---")
    for i, m in enumerate(expert_pool):
        print(f"{m}: {base_acc_by_model[i]:.2f}%")
        
    # 执行 Bench-Harness
    print("\n--- Running Bench-Harness Evaluation ---")
    correct_count = 0
    total_count = len(test_queries)
    
    for i in range(total_count):
        q = test_queries[i]
        
        # 路由选择
        exec_idx, rev_idx = engine.route_query(
            q, K, args.lam, args.alpha, args.beta, args.gamma
        )
        
        # 简单系统评判模拟：
        # 执行者答对 = 团队大概率可以通过（需要审查者不误导/FDR机制保障）
        # 执行者答错 + 审查者答对 = 成功救援
        m_exec_correct = test_Y[i, exec_idx] == 1
        m_rev_correct = test_Y[i, rev_idx] == 1
        
        # 真实环境中的多智能体对话博弈结果很难单纯靠静态矩阵全模拟。
        # 理论近似评估（UpperBound）：如果执行者对且审查者不反对(或审查者也对)，或者执行者错但审查者对并纠正。
        # 为了验证数学体系的性能潜能，这里取 `Union` 容错率 (如果两个人有一个对则算通过，模拟完美说服):
        team_correct = m_exec_correct or m_rev_correct
        if team_correct:
            correct_count += 1
            
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{total_count}...")
            
    final_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n[Bench-Harness Result]: {final_acc:.2f}%")
    print("\n结论：你可以对比单模型的 Accuracy 和 Bench-Harness Team 的 Accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bench-Harness Evaluation on GAOKAO")
    parser.add_argument("--data_root", type=str, default=".", help="Workspace path to GAOKAO-Bench directories")
    parser.add_argument("--models", type=str, default="Qwen-72B-Chat,glm-4-9b-chat,gemma-2-9b-it", help="Comma separated list of expert models")
    parser.add_argument("--subjects", type=str, default="Physics_MCQs,Math_I_MCQs,Chemistry_MCQs", help="Comma separated subjects")
    parser.add_argument("--encoder", type=str, default="BAAI/bge-m3", help="Dense retrieval model for NP-LOM")
    parser.add_argument("--test_year", type=str, default="2010-2022", help="OOD Test years (e.g., 2010-2022)")
    
    # Bench-Harness Hyperparameters
    parser.add_argument("--top_k", type=float, default=0.2, help="K for local manifold. If <1.0, acts as ratio of calib set.")
    parser.add_argument("--lam", type=float, default=5.0, help="Bayesian smoothing factor lambda")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Smoothed CRR")
    parser.add_argument("--beta", type=float, default=0.8, help="Weight for Smoothed FDR")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight for Competence Anchor (Accuracy)")
    
    args = parser.parse_args()
    evaluate_pipeline(args)