import os, json, argparse, math, numpy as np
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 复用 run2_mmlupro.py 中的头文件函数
from run2_mmlupro import parse_args, load_local_results

def main():
    args = parse_args()
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    ref_data_full, test_data_full = list(dataset['validation']), list(dataset['test'])

    mapped_subjects = [s.lower().replace("_", " ") for s in args.subjects]
    ref_data = [item for item in ref_data_full if item['category'].lower() in mapped_subjects]
    test_data = [item for item in test_data_full if item['category'].lower() in mapped_subjects]
    if not test_data: ref_data, test_data = ref_data_full, test_data_full

    embedder = SentenceTransformer(args.embed_model)
    ref_embs = embedder.encode([item['question'] for item in ref_data], show_progress_bar=False)
    test_embs = embedder.encode([item['question'] for item in test_data], show_progress_bar=False)
    ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)

    model_preds, model_correct = load_local_results(args.results_dir, args.models)

    use_same_set = False
    ref_has_results = any(str(ref_data[0].get('question_id', ref_data[0].get('question', ''))) in model_correct[m] for m in args.models) if ref_data else False
    if not ref_has_results: ref_data, ref_embs, use_same_set = test_data, test_embs, True

    correct_count, total_models_used = 0, 0
    sims_matrix = np.dot(test_embs, ref_embs.T)

    print("Running MMLU-Pro Tri-node Cascade Routing...")
    for i, item in enumerate(test_data):
        qid = str(item.get('question_id', item.get('question', '')))
        sims = sims_matrix[i].copy()
        if use_same_set: sims[i] = -1.0
        
        k = max(1, int(len(sims) * args.top_k_ratio))
        top_indices = np.argsort(sims)[-k:]
        
        scores = {}
        for m in args.models:
            score = args.alpha
            for idx in top_indices:
                sim = sims[idx]
                weight = (max(sim, 0) ** args.gamma) * math.exp(sim / args.lambda_smooth)
                ref_qid = str(ref_data[idx].get('question_id', ref_data[idx].get('question', '')))
                if model_correct[m].get(ref_qid, False): score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        models_used, final_answer, exited = 0, "A", False
        for batch_start in range(0, len(models_sorted), 3):
            trinode_batch = models_sorted[batch_start:batch_start+3]
            batch_votes = []
            
            for m in trinode_batch:
                batch_votes.append(model_preds[m].get(qid, "A"))
                models_used += 1
                
            counts = Counter(batch_votes)
            most_common_pred, most_common_count = counts.most_common(1)[0]
            
            if (most_common_count / len(batch_votes)) >= args.early_exit_threshold:
                final_answer = most_common_pred
                exited = True
                break
                
        if not exited:
            all_votes = [model_preds[m].get(qid, "A") for m in models_sorted[:models_used]]
            final_answer = Counter(all_votes).most_common(1)[0][0]
            
        if final_answer == item['answer']: correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"MMLU-Pro Tri-node Routing Acc: {correct_count / len(test_data) * 100:.2f}% | Avg Models: {total_models_used / len(test_data):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()