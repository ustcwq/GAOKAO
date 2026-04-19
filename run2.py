import os
import json
import argparse
import math
import numpy as np
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="MMLU-Pro Entropy-based Routing Harness")
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--embed_model', type=str, default="models/bge-m3")
    parser.add_argument('--top_k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_smooth', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--early_exit_threshold', type=float, default=0.7)
    # 【修复重点】：默认路径改为 MMLU-Pro/results
    parser.add_argument('--results_dir', type=str, default="MMLU-Pro/results")
    return parser.parse_args()

def load_local_results(results_dir, models):
    model_preds, model_correct = {}, {}
    for m in models:
        model_preds[m], model_correct[m] = {}, {}
        # 兼容读取 MMLU-Pro 常见的评估结果命名
        paths = [
            os.path.join(results_dir, f"{m}_result.json"),
            os.path.join(results_dir, m, "eval_results.json"),
            os.path.join(results_dir, f"{m}.json")
        ]
        
        # 增加深度搜索匹配，容忍文件名大小写和存放子目录位置的差异
        if os.path.exists(results_dir):
            for root, _, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.json') and m.lower() in file.lower():
                        paths.append(os.path.join(root, file))
        
        # 去除重复搜索的路径
        paths = list(dict.fromkeys(paths))
        
        data = None
        loaded_path = ""
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    loaded_path = p
                    break # 成功加载后立即跳出
                except json.JSONDecodeError:
                    continue
        
        if data:
            print(f"✅ [成功] 找到并加载模型 {m} 结果 -> {loaded_path}")
            # MMLU-Pro 某些版本生成的结果可能包裹在 dict 键里
            if isinstance(data, dict) and "results" in data:
                data = data["results"]
                
            for item in data:
                # 兼容不同结构下的 question 或 question_id 键名
                qid = str(item.get('question_id', item.get('question', '')))
                pred = item.get('pred', 'A')
                ans = item.get('answer', '')
                model_preds[m][qid] = pred
                model_correct[m][qid] = (pred == ans)
        else:
            print(f"❌ [警告] 未在 {results_dir} 找到模型 {m} 的结果。")
            
    return model_preds, model_correct

def main():
    args = parse_args()
    
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_data = dataset['test']

    # MMLU-Pro 学科映射处理
    mmlu_pro_categories = set([item['category'] for item in test_data])
    mapped_subjects = [s.lower().replace("_", " ") for s in args.subjects]
    valid_subjects = [s for s in mapped_subjects if s in mmlu_pro_categories]
    
    filtered_data = [item for item in test_data if item['category'].lower() in valid_subjects]
    if not filtered_data:
        print("⚠️ [警告] 传入科目与 MMLU-Pro 不匹配，降级为测试所有科目。")
        filtered_data = list(test_data)

    print(f"Loading embedding model: {args.embed_model}...")
    embedder = SentenceTransformer(args.embed_model)
    questions = [item['question'] for item in filtered_data]
    test_embs = embedder.encode(questions, show_progress_bar=True)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    
    print("\n" + "-"*40)
    print("开始搜索并加载离线模型结果...")
    model_preds, model_correct = load_local_results(args.results_dir, args.models)
    print("-" * 40 + "\n")

    correct_count = 0
    total_models_used = 0
    early_exit_count = 0

    print("Running Entropy-based Dynamic Routing...")
    for i, item in enumerate(filtered_data):
        qid = str(item.get('question_id', item.get('question', '')))
        query_emb = test_embs[i]
        
        # 1. 计算相似度并获取 Top-K 参考题目
        sims = np.dot(test_embs, query_emb)
        sims[i] = -1.0 # 排除当前题目自身
        k = max(1, int(len(sims) * args.top_k_ratio))
        top_indices = np.argsort(sims)[-k:]
        
        # 2. 计算模型动态路由得分
        scores = {}
        for m in args.models:
            score = args.alpha
            for idx in top_indices:
                sim = sims[idx]
                weight = (max(sim, 0) ** args.gamma) * math.exp(sim / args.lambda_smooth)
                ref_qid = str(filtered_data[idx].get('question_id', filtered_data[idx].get('question', '')))
                if model_correct[m].get(ref_qid, False):
                    score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 3. 创新二：信息熵计算与早退逻辑
        votes = []
        models_used = 0
        final_answer = "A"
        
        for m in models_sorted:
            # 如果某模型对于这道题没有预测，则默认选 A
            pred = model_preds[m].get(qid, "A")
            votes.append(pred)
            models_used += 1
            
            # 最少调用2个模型才开始计算分布熵
            if len(votes) >= 2:
                counts = Counter(votes)
                probs = [c / len(votes) for c in counts.values()]
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                
                if entropy < args.early_exit_threshold:
                    final_answer = counts.most_common(1)[0][0]
                    early_exit_count += 1
                    break
        else:
            final_answer = Counter(votes).most_common(1)[0][0]
            
        if final_answer == item['answer']:
            correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"MMLU-Pro Entropy Routing Accuracy: {correct_count / len(filtered_data) * 100:.2f}%")
    print(f"Early Exit Triggered: {early_exit_count} / {len(filtered_data)} questions")
    print(f"Average Models Queried: {total_models_used / len(filtered_data):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()