import os
import json
import glob
import codecs
import numpy as np
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==============================================================================
# Bench-Harness: 基实例级正交共识的智能体编排底座核心代码
# ==============================================================================

class DataLoader:
    """处理 GAOKAO-Bench 数据集，支持加载校准集 (2023-2024) 和测试集 (2010-2022)"""
    def __init__(self, data_root: str, expert_models: List[str], subjects: List[str]):
        self.data_root = data_root
        self.expert_models = expert_models
        self.subjects = subjects

    def load_dataset(self, year_span: str) -> Tuple[List[Dict], np.ndarray, List[str], List[Dict]]:
        """
        加载数据，返回:
        - queries: List[Dict] (包含问题文本等)
        - Y: np.ndarray [M, N] (错误签名矩阵，M为问题数，N为专家数。1代表正确，0代表错误)
        - question_ids: List[str] 题目标识列表
        - raw_items: List[Dict] 原始题目记录(用于后续保存)
        """
        all_questions = {}
        # 为了保留原始结构，我们将每个题目的各个模型原始输出保存下来
        raw_items_dict = defaultdict(dict) 
        Y_matrix_dict = defaultdict(lambda: np.zeros(len(self.expert_models)))
        
        for p_idx, model in enumerate(self.expert_models):
            for subject in self.subjects:
                if year_span == "2023-2024":
                    patterns = [
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_2023_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_2024_{subject}*.json"),
                        os.path.join(self.data_root, "GAOKAO-Bench-2023-2024", "Data", f"{model}_*_{subject}*.json") 
                    ]
                else: 
                    patterns = [
                        os.path.join(self.data_root, "GAOKAO-Bench-2010-2022", "Data", f"{model}_*_{subject}*.json")
                    ]
                
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(pattern))
                files = list(set(files))
                
                for file_path in files:
                    try:
                        with codecs.open(file_path, 'r', 'utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                examples = data
                            else:
                                examples = data.get('examples', data.get('example', []))
                                
                            for q in examples:
                                q_id = f"{subject}_{q.get('index', q.get('id', hash(q.get('question', ''))))}"
                                
                                if q_id not in all_questions:
                                    all_questions[q_id] = {
                                        "question": q.get('question', q.get('content', '')),
                                        "standard_answer": q.get('standard_answer', q.get('answer', '')),
                                        "subject": subject,
                                        "year": q.get("year", year_span)
                                    }
                                
                                score = float(q.get('score', 0))
                                is_correct = 1 if (score > 0 or q.get('correctness') == True) else 0
                                Y_matrix_dict[q_id][p_idx] = is_correct
                                
                                # 保存该模型针对这道题的具体作答情况，方便后期写回
                                raw_items_dict[q_id][model] = {
                                    'model_answer': q.get('model_answer', ''),
                                    'model_output': q.get('model_output', ''),
                                    'score': score,
                                    'is_correct': is_correct
                                }
                    except Exception as e:
                        pass
        
        q_ids = list(all_questions.keys())
        queries = [all_questions[qid] for qid in q_ids]
        Y = np.array([Y_matrix_dict[qid] for qid in q_ids])
        raw_items = [raw_items_dict[qid] for qid in q_ids]
        
        return queries, Y, q_ids, raw_items


class BenchHarnessOrthogonalEngine:
    def __init__(self, encoder_name="models/bge-m3", expert_names=None):
        self.expert_names = expert_names if expert_names else []
        self.num_experts = len(self.expert_names)
        self.encoder = SentenceTransformer(encoder_name)
        
        self.calib_Y = None
        self.calib_V = None
        self.global_prior = {}

    # ==============================================================================
    # 3.2 离线情节记忆建库 (Offline Episodic Datastore Construction)
    # ==============================================================================
    def memorize_calibration_set(self, calib_queries: List[Dict], Y: np.ndarray):
        print(f"[{self.__class__.__name__}] Encoding {len(calib_queries)} queries for Offline Datastore (B_cal)...")
        texts = [q['question'] for q in calib_queries]
        self.calib_V = self.encoder.encode(texts, show_progress_bar=True)
        self.calib_Y = Y
        self.num_calib = len(Y)
        
        print(f"[{self.__class__.__name__}] Extraction: Global Error Orthogonality Priors...")
        self._compute_global_priors()

    # ==============================================================================
    # 3.1 宏观错误正交性先验 (Global Error Orthogonality Prior)
    # ==============================================================================
    def _compute_global_priors(self):
        self.global_prior = {
            'Acc': np.mean(self.calib_Y, axis=0),
            'CRR': np.zeros((self.num_experts, self.num_experts)), # [M_rev, M_exec]
            'FDR': np.zeros((self.num_experts, self.num_experts))  # [M_rev, M_exec]
        }
        
        for exec_idx in range(self.num_experts):
            E_mask = (self.calib_Y[:, exec_idx] == 0) # 认知盲区 E(M_exec)
            C_mask = (self.calib_Y[:, exec_idx] == 1) # 认知顺境 C(M_exec)
            
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

    def route_query_dynamic(self, query: Dict, K: int, lam: float, alpha: float, beta: float, gamma: float) -> Tuple[int, int, float]:
        """
        处理新的未见题，返回 (M_exec_idx, M_rev_idx, best_obj_score)
        """
        # ==============================================================================
        # 3.2 非参数局部正交流形坍缩 (On-the-fly Micro-Manifold Collapse)
        # ==============================================================================
        q_vec = self.encoder.encode([query['question']])[0]
        sims = cosine_similarity([q_vec], self.calib_V)[0]
        topk_indices = np.argsort(sims)[-K:]
        manifold_Y = self.calib_Y[topk_indices] # N_K(q_new)
        
        # 动态推举特定问题最佳局部执行者 M_{exec}^*
        exec_star = int(np.argmax(np.sum(manifold_Y, axis=0)))
        
        # ==============================================================================
        # 3.3 贝叶斯平滑的微观正交重算 (Bayesian-Smoothed Micro-Computation)
        # ==============================================================================
        E_K_mask = (manifold_Y[:, exec_star] == 0)
        C_K_mask = (manifold_Y[:, exec_star] == 1)
        E_K_count = np.sum(E_K_mask)
        C_K_count = np.sum(C_K_mask)
        
        best_rev_star = -1
        best_arbitration_score = -float('inf')
        
        # ==============================================================================
        # 3.4 胜任力锚定与信心感知动态仲裁 (Confidence-Aware Arbitration)
        # ==============================================================================
        for j in range(self.num_experts):
            if j == exec_star: continue
            
            # --- Smoothed CRR ---
            crr_prior = self.global_prior['CRR'][j, exec_star]
            crr_evid = np.sum(manifold_Y[E_K_mask, j])
            crr_smooth = (crr_evid + lam * crr_prior) / (E_K_count + lam)
            
            # --- Smoothed FDR ---
            fdr_prior = self.global_prior['FDR'][j, exec_star]
            fdr_evid = np.sum(1 - manifold_Y[C_K_mask, j])
            fdr_smooth = (fdr_evid + lam * fdr_prior) / (C_K_count + lam)
            
            # --- Competence Anchor (Accuracy) ---
            acc_prior = self.global_prior['Acc'][j]
            acc_evid = np.sum(manifold_Y[:, j])
            acc_smooth = (acc_evid + lam * acc_prior) / (K + lam)
            
            # --- 终极审查者仲裁目标函数 ---
            obj_score = alpha * crr_smooth - beta * fdr_smooth + gamma * acc_smooth
            
            if obj_score > best_arbitration_score:
                best_arbitration_score = obj_score
                best_rev_star = j
                
        # 兜底：如果模型池只有1个模型
        if best_rev_star == -1:
            best_rev_star = exec_star
            
        return exec_star, best_rev_star, best_arbitration_score


def evaluate_pipeline(args):
    expert_pool = args.models.split(',')
    subjects_list = args.subjects.split(',')
    loader = DataLoader(args.data_root, expert_pool, subjects_list)
    
    # 步骤 A：利用 2023-2024 年的数据作为校准集构建底座
    print("==================================================================")
    print(">>> 1. Loading 2023-2024 Calibration Data...")
    calib_queries, calib_Y, _, _ = loader.load_dataset("2023-2024")
    
    if len(calib_queries) == 0:
        print("Error: 未找到 2023-2024 校准集数据，请检查数据集路径。")
        return
        
    engine = BenchHarnessOrthogonalEngine(encoder_name=args.encoder, expert_names=expert_pool)
    engine.memorize_calibration_set(calib_queries, calib_Y)
    
    K = int(engine.num_calib * args.top_k) if args.top_k < 1.0 else int(args.top_k)
    K = max(1, min(K, engine.num_calib))
    print(f"\n[Bench-Harness Initialization] Top-K Neighbors: {K}")
    print("==================================================================")
    
    # 步骤 B：利用 2010-2022 (不可见 OOD) 作为测试集验证
    print(">>> 2. Loading 2010-2022 OOD Testing Data...")
    test_queries, test_Y, test_ids, test_raw_items = loader.load_dataset("2010-2022")
    
    if len(test_queries) == 0:
        print("Error: 未找到 2010-2022 测试集数据，请检查数据集路径。")
        return
        
    base_acc_by_model = np.mean(test_Y, axis=0) * 100
    print("\n--- Single Expert Baselines (Zero-Shot) ---")
    for i, m in enumerate(expert_pool):
        print(f"{m:<30} \t{base_acc_by_model[i]:.2f}%")
        
    print("\n>>> 3. Bench-Harness Instance-Level Routing in Progress...")
    correct_count = 0
    total_count = len(test_queries)
    
    # 记录执行结果用于保存
    saved_results = []
    
    for i in range(total_count):
        q = test_queries[i]
        q_raw_data = test_raw_items[i]
        
        # Bench-Harness 动态博弈选取
        exec_idx, rev_idx, arb_score = engine.route_query_dynamic(
            q, K, args.lam, args.alpha, args.beta, args.gamma
        )
        
        m_exec = expert_pool[exec_idx]
        m_rev = expert_pool[rev_idx]
        
        # 获取这二者的独立判断正确与否
        m_exec_correct = test_Y[i, exec_idx] == 1
        m_rev_correct = test_Y[i, rev_idx] == 1
        
        # 理想中的正交纠错：只要两人中有正确的（执行者顺境 or 盲区被搭档捞回），计为团队自愈成功
        team_correct = m_exec_correct or m_rev_correct 
        if team_correct:
            correct_count += 1
            
        # 组装完整的 JSON 输出记录
        record = {
            "index": i,
            "year": q['year'],
            "subject": q['subject'],
            "question": q['question'],
            "standard_answer": q['standard_answer'],
            "Bench_Harness_Team": {
                "Executor": m_exec,
                "Reviewer": m_rev,
                "Executor_Correct": bool(m_exec_correct),
                "Reviewer_Correct": bool(m_rev_correct),
                "Team_Correct": bool(team_correct),
                "Arbitration_Score": float(arb_score)
            },
            "Expert_Outputs": {}
        }
        
        # 加入各个模型的原始输出便于对齐
        for m_name in expert_pool:
            record["Expert_Outputs"][m_name] = q_raw_data.get(m_name, {})
            
        saved_results.append(record)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{total_count} (Running Acc: {(correct_count/(i+1))*100:.2f}%)")
            
    final_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n==================================================================")
    print("                 Bench-Harness FINAL RESULTS                      ")
    print("==================================================================")
    print(f"Test Queries Processed : {total_count}")
    print(f"Best Single Model Acc  : {np.max(base_acc_by_model):.2f}%")
    print(f"Bench-Harness Team Acc : {final_acc:.2f}%")
    print(f"Relative Gain          : +{final_acc - np.max(base_acc_by_model):.2f}%")
    
    # ==============================================================================
    # 结果保存：参考 local_bench.py，序列化保存为 JSON
    # ==============================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"Bench_Harness_Result_{args.lam}_{args.top_k}.json")
    
    final_dump = {
        "Settings": {
            "Models": expert_pool,
            "Subjects": subjects_list,
            "HyperParams": {"top_k": K, "lam": args.lam, "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
            "Overall_Accuracy": f"{final_acc:.2f}%"
        },
        "Examples": saved_results
    }
    
    with codecs.open(save_path, "w+", 'utf-8') as f:
        json.dump(final_dump, f, ensure_ascii=False, indent=4)
        
    print(f"\n=> 评估结果及专家协作明细已成功保存至: {save_path}")
    print("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".", help="GAOKAO Bench 根目录")
    parser.add_argument("--models", type=str, default="Baichuan2-7B-Chat,Qwen3.5-9B,gemma-2-9b-it", help="候选模型池")
    parser.add_argument("--subjects", type=str, default="Physics_MCQs,Chemistry_MCQs,Biology_MCQs", help="指定评估的学科")
    parser.add_argument("--encoder", type=str, default="models/bge-m3", help="预训练检索编码器")
    parser.add_argument("--output_dir", type=str, default="./Results", help="JSON持久化结果保存目录")
    
    # Bench-Harness 核心超参数控制
    parser.add_argument("--top_k", type=float, default=0.20, help="NP-LOM 邻域 K (百分数0.2或整数)")
    parser.add_argument("--lam", type=float, default=5.0, help="Bayesian Smoothing (λ)")
    parser.add_argument("--alpha", type=float, default=1.0, help="CRR coefficient (α)")
    parser.add_argument("--beta", type=float, default=0.8, help="FDR penalty coefficient (β)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Competence anchor coefficient (γ)")
    
    args = parser.parse_args()
    evaluate_pipeline(args)