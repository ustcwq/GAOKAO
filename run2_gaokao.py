import os
import json
import argparse
import math
import glob
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="MMLU-Pro Entropy Routing (Ref: GAOKAO 2022-2023)")
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--embed_model', type=str, default="models/bge-m3")
    parser.add_argument('--top_k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_smooth', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--early_exit_threshold', type=float, default=0.7)
    
    # 自动定位两边的数据与结果路径
    parser.add_argument('--gaokao_data_dir', type=str, default="data")
    parser.add_argument('--gaokao_results_dir', type=str, default="results")
    parser.add_argument('--mmlupro_results_dir', type=str, default="MMLU-Pro/results")
    return parser.parse_args()

def load_local_results(results_dir, models):
    model_preds, model_correct = {}, {}
    for m in models:
        model_preds[m], model_correct[m] = {}, {}
        paths = []
        if os.path.exists(results_dir):
            for root, _, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.json') and m.lower() in file.lower():
                        paths.append(os.path.join(root, file))
        for p in paths:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = []
                if isinstance(data, dict) and "results" in data:
                    items = data["results"]
                elif isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list): items.extend(v)
                        elif isinstance(v, dict): items.append(v)
                
                for item in items:
                    qid = str(item.get('question_id', item.get('id', '')))
                    qtext = str(item.get('question', item.get('instruction', ''))).strip()
                    pred = str(item.get('pred', item.get('prediction', 'A')))
                    ans = str(item.get('answer', ''))
                    
                    is_correct = (pred == ans)
                    if qid:
                        model_preds[m][qid] = pred
                        model_correct[m][qid] = is_correct
                    if qtext:
                        model_preds[m][qtext] = pred
                        model_correct[m][qtext] = is_correct
            except: continue
    return model_preds, model_correct

def load_gaokao_reference(data_dir):
    ref_data = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            items = data.get("example", data) if isinstance(data, dict) else data
                            if not isinstance(items, list): continue
                            for item in items:
                                year = str(item.get("year", ""))
                                # 严格筛选: 只选取 2022-2023 年的高考题作为参考池
                                if "2022" in year or "2023" in year or "2022" in file or "2023" in file:
                                    if 'question' in item:
                                        ref_data.append(item)
                    except: pass
    return ref_data

def main():
    args = parse_args()
    print("="*60)
    print("实验一：目标测试集 = MMLU-Pro Test | 相似度参考池 = GAOKAO 2022-2023")
    print("="*60)

    print("1. 加载 MMLU-Pro 测试集...")
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        test_data = list(dataset['test'])
    except Exception:
        print("HuggingFace 连通失败，启用本地 fallback (MMLU-Pro/data)...")
        test_files = glob.glob("MMLU-Pro/data/*test*.parquet")
        test_data = pd.concat([pd.read_parquet(f) for f in test_files]).to_dict('records') if test_files else []

    print("2. 加载 GAOKAO (2022-2023) 参考题库...")
    ref_data = load_gaokao_reference(args.gaokao_data_dir)
    print(f"   -> 成功加载高考参考题目: {len(ref_data)} 道")

    mmlu_pro_categories = set([item.get('category', 'unknown') for item in test_data])
    mapped_subjects = [s.lower().replace("_", " ") for s in args.subjects]
    valid_subjects = [s for s in mapped_subjects if s in mmlu_pro_categories]
    filtered_test_data = [item for item in test_data if item.get('category', 'unknown').lower() in valid_subjects]
    if not filtered_test_data:
        print("⚠️ 传入科目与 MMLU-Pro 不匹配，降级测试全量数据。")
        filtered_test_data = test_data

    print(f"3. 正在加载 Embedding 模型: {args.embed_model} ...")
    embedder = SentenceTransformer(args.embed_model)
    print("   -> 正在编码 GAOKAO 参考池...")
    ref_embs = embedder.encode([str(item.get('question', '')) for item in ref_data], show_progress_bar=True)
    ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
    
    print("   -> 正在编码 MMLU-Pro 测试集...")
    test_embs = embedder.encode([str(item.get('question', '')) for item in filtered_test_data], show_progress_bar=True)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)

    print("4. 加载离线模型评测结果...")
    test_preds, _ = load_local_results(args.mmlupro_results_dir, args.models)
    _, ref_correct = load_local_results(args.gaokao_results_dir, args.models)

    correct_count, total_models_used, early_exit_count = 0, 0, 0

    print("5. 启动跨域动态路由推断...")
    for i, test_item in enumerate(filtered_test_data):
        test_qid = str(test_item.get('question_id', ''))
        test_qtext = str(test_item.get('question', '')).strip()
        query_emb = test_embs[i]
        
        # 直接计算全量跨域相似度（无需排除自身 sims[i] = -1.0）
        sims = np.dot(ref_embs, query_emb)
        k = max(1, int(len(sims) * args.top_k_ratio))
        top_indices = np.argsort(sims)[-k:]
        
        scores = {}
        for m in args.models:
            score = args.alpha
            for idx in top_indices:
                sim = sims[idx]
                weight = (max(sim, 0) ** args.gamma) * math.exp(sim / args.lambda_smooth)
                ref_item = ref_data[idx]
                ref_qid = str(ref_item.get('question_id', ''))
                ref_qtext = str(ref_item.get('question', '')).strip()
                
                # 检查此大模型是否在这个高考参考题目上得分
                is_correct = ref_correct[m].get(ref_qid, ref_correct[m].get(ref_qtext, False))
                if is_correct: score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 创新二：信息熵早退
        votes, models_used, final_answer = [], 0, "A"
        for m in models_sorted:
            pred = test_preds[m].get(test_qid, test_preds[m].get(test_qtext, "A"))
            votes.append(pred)
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
            
        if final_answer == str(test_item.get('answer', '')):
            correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"跨域路由 (Ref: GAOKAO -> Test: MMLU-Pro) 准确率: {correct_count / len(filtered_test_data) * 100:.2f}%")
    print(f"Early Exit Triggered: {early_exit_count} / {len(filtered_test_data)}")
    print(f"Average Models Queried: {total_models_used / len(filtered_test_data):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()