import os
import json
import argparse
import math
import numpy as np
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="MMLU-Pro Tri-node Graph Routing Harness")
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--embed_model', type=str, default="models/bge-m3")
    parser.add_argument('--top_k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_smooth', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--early_exit_threshold', type=float, default=0.7)
    parser.add_argument('--results_dir', type=str, default="results")
    return parser.parse_args()

# 直接复用上方文件中的 load_local_results 逻辑
from run_bench_harness_entropy import load_local_results

def main():
    args = parse_args()
    
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_data = dataset['test']

    mmlu_pro_categories = set([item['category'] for item in test_data])
    mapped_subjects = [s.lower().replace("_", " ") for s in args.subjects]
    valid_subjects = [s for s in mapped_subjects if s in mmlu_pro_categories]
    
    filtered_data = [item for item in test_data if item['category'].lower() in valid_subjects]
    if not filtered_data: filtered_data = list(test_data)

    print(f"Loading embedding model: {args.embed_model}...")
    embedder = SentenceTransformer(args.embed_model)
    test_embs = embedder.encode([item['question'] for item in filtered_data], show_progress_bar=False)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    
    model_preds, model_correct = load_local_results(args.results_dir, args.models)

    correct_count = 0
    total_models_used = 0

    print("Running Tri-node Cascade Routing Evaluation...")
    for i, item in enumerate(filtered_data):
        qid = item.get('question_id', item.get('question', ''))
        query_emb = test_embs[i]
        
        sims = np.dot(test_embs, query_emb)
        sims[i] = -1.0 
        k = max(1, int(len(sims) * args.top_k_ratio))
        top_indices = np.argsort(sims)[-k:]
        
        scores = {}
        for m in args.models:
            score = args.alpha
            for idx in top_indices:
                sim = sims[idx]
                weight = (max(sim, 0) ** args.gamma) * math.exp(sim / args.lambda_smooth)
                ref_qid = filtered_data[idx].get('question_id', filtered_data[idx].get('question', ''))
                if model_correct[m].get(ref_qid, False): 
                    score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 创新三：Tri-node 架构，将模型切分为步长为 3 的阵列节点
        models_used = 0
        final_answer = "A"
        exited = False
        
        for batch_start in range(0, len(models_sorted), 3):
            trinode_batch = models_sorted[batch_start:batch_start+3]
            batch_votes = []
            
            for m in trinode_batch:
                pred = model_preds[m].get(qid, "A")
                batch_votes.append(pred)
                models_used += 1
            
            # 计算这一批 Tri-node 内的共识 (Consensus)
            counts = Counter(batch_votes)
            most_common_pred, most_common_count = counts.most_common(1)[0]
            
            # 若 Tri-node 节点组内的共识概率达到了设定阈值，即刻阻断长尾下发
            if (most_common_count / len(batch_votes)) >= args.early_exit_threshold:
                final_answer = most_common_pred
                exited = True
                break
                
        if not exited:
            # 走到最后若无共识，依然取全局最大票数
            all_votes = [model_preds[m].get(qid, "A") for m in models_sorted[:models_used]]
            final_answer = Counter(all_votes).most_common(1)[0][0]
            
        if final_answer == item['answer']:
            correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"MMLU-Pro Trinode Routing Accuracy: {correct_count / len(filtered_data) * 100:.2f}%")
    print(f"Average Models Queried Per Question: {total_models_used / len(filtered_data):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()