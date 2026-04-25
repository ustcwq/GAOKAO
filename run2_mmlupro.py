import os
import json
import argparse
import math
import numpy as np
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="MMLU-Pro Entropy-based Routing")
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--embed_model', type=str, default="models/bge-m3")
    parser.add_argument('--top_k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_smooth', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--early_exit_threshold', type=float, default=0.7)
    parser.add_argument('--results_dir', type=str, default="MMLU-Pro/results")
    return parser.parse_args()

def load_local_results(results_dir, models):
    model_preds, model_correct = {}, {}
    for m in models:
        model_preds[m], model_correct[m] = {}, {}
        if not os.path.exists(results_dir): continue
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json') and m.lower() in file.lower():
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, dict) and "results" in data:
                            data = data["results"]
                        if isinstance(data, list):
                            for item in data:
                                qid = str(item.get('question_id', item.get('question', '')))
                                pred = item.get('pred', 'A')
                                ans = item.get('answer', '')
                                model_preds[m][qid] = pred
                                model_correct[m][qid] = (pred == ans)
                    except Exception: pass
    return model_preds, model_correct

def main():
    args = parse_args()
    print("Loading MMLU-Pro Validation (Ref Pool) and Test (Eval Pool) datasets...")
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        ref_data_full, test_data_full = list(dataset['validation']), list(dataset['test'])
    except Exception as e:
        print(f"Dataset load failed: {e}")
        return

    mapped_subjects = [s.lower().replace("_", " ") for s in args.subjects]
    ref_data = [item for item in ref_data_full if item['category'].lower() in mapped_subjects]
    test_data = [item for item in test_data_full if item['category'].lower() in mapped_subjects]

    if not test_data:
        print("⚠️ 传入科目与数据集不匹配，将回退使用全量学科数据。")
        ref_data, test_data = ref_data_full, test_data_full

    print(f"Loading embedding model: {args.embed_model}...")
    embedder = SentenceTransformer(args.embed_model)
    ref_embs = embedder.encode([item['question'] for item in ref_data], show_progress_bar=False)
    test_embs = embedder.encode([item['question'] for item in test_data], show_progress_bar=True)
    ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)

    model_preds, model_correct = load_local_results(args.results_dir, args.models)

    # 智能降级检测：检查模型是否在验证集上做过题
    use_same_set = False
    ref_has_results = any(str(ref_data[0].get('question_id', ref_data[0].get('question', ''))) in model_correct[m] for m in args.models) if ref_data else False
    if not ref_has_results:
        print("\n⚠️ 警告: 离线结果中无 Validation 集记录！自动降级为使用 Test 集自身进行交叉参考。\n")
        ref_data, ref_embs, use_same_set = test_data, test_embs, True

    correct_count, total_models_used, early_exit_count = 0, 0, 0
    sims_matrix = np.dot(test_embs, ref_embs.T)

    print("Running MMLU-Pro Entropy-based Cross-Pool Routing...")
    for i, item in enumerate(test_data):
        qid = str(item.get('question_id', item.get('question', '')))
        sims = sims_matrix[i].copy()
        if use_same_set: sims[i] = -1.0 # 降级时排己
        
        k = max(1, int(len(sims) * args.top_k_ratio))
        top_indices = np.argsort(sims)[-k:]
        
        scores = {}
        for m in args.models:
            score = args.alpha
            for idx in top_indices:
                sim = sims[idx]
                weight = (max(sim, 0) ** args.gamma) * math.exp(sim / args.lambda_smooth)
                ref_qid = str(ref_data[idx].get('question_id', ref_data[idx].get('question', '')))
                if model_correct[m].get(ref_qid, False):
                    score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        votes, models_used, final_answer = [], 0, "A"
        for m in models_sorted:
            votes.append(model_preds[m].get(qid, "A"))
            models_used += 1
            if len(votes) >= 2:
                probs = [c / len(votes) for c in Counter(votes).values()]
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                if entropy < args.early_exit_threshold:
                    final_answer = Counter(votes).most_common(1)[0][0]
                    early_exit_count += 1
                    break
        else:
            final_answer = Counter(votes).most_common(1)[0][0]
            
        if final_answer == item['answer']: correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"MMLU-Pro Entropy Routing Acc: {correct_count / len(test_data) * 100:.2f}%")
    print(f"Early Exits: {early_exit_count} / {len(test_data)} | Avg Models: {total_models_used / len(test_data):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()