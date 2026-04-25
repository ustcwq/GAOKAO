import os, json, argparse, math, numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--embed_model', type=str, default="models/bge-m3")
    parser.add_argument('--top_k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_smooth', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--early_exit_threshold', type=float, default=0.7)
    parser.add_argument('--ref_data_dir', type=str, default="GAOKAO-Bench-2023-2024/Data/GAOKAO-Bench-2023")
    parser.add_argument('--test_data_dir', type=str, default="GAOKAO-Bench-2023-2024/Data/GAOKAO-Bench-2024")
    parser.add_argument('--results_dir', type=str, default="GAOKAO-Bench-2023-2024/Data")
    return parser.parse_args()

def load_gaokao_data(data_dir, subjects, models):
    data = []
    if not os.path.exists(data_dir): return data
    for root, _, files in os.walk(data_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if not file.endswith('.json'): continue
            # 防止把模型预测结果当成题库加载
            if any(m.lower() in filepath.lower() for m in models): continue
            # 学科匹配（忽略大小写以增加鲁棒性）
            if not any(sub.lower() in file.lower() for sub in subjects): continue
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                items = content.get("example", content) if isinstance(content, dict) else content
                if not isinstance(items, list): continue
                for item in items:
                    if not isinstance(item, dict): continue
                    q_text = str(item.get('question', item.get('instruction', ''))).strip()
                    qid = str(item.get('index', item.get('id', '')))
                    ans = str(item.get('answer', item.get('standard_answer', 'A')))
                    if q_text: data.append({'question_id': qid, 'question': q_text, 'answer': ans})
            except: continue
    return data

def load_local_results(results_dir, models):
    model_preds, model_correct = {}, {}
    print(f"\n🔍 开始从 {results_dir} 深度扫描 GAOKAO 离线预测结果...")
    for m in models:
        model_preds[m], model_correct[m] = {}, {}
        loaded_cnt = 0
        if not os.path.exists(results_dir): continue
        for root, _, files in os.walk(results_dir):
            for file in files:
                filepath = os.path.join(root, file)
                # 【重大修复】：移除了对根目录的屏蔽！只要路径里含模型名就尝试读取
                if m.lower() in filepath.lower() and not os.path.isdir(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 安全锁：如果是原始题库（没有results字典），直接跳过防止误读
                        if isinstance(data, dict) and "example" in data and "results" not in data:
                            continue
                            
                        iterable = []
                        if isinstance(data, dict):
                            iterable = data.get("results", data.get("eval_results", data.get("example", list(data.values()))))
                        elif isinstance(data, list):
                            iterable = data
                            
                        if not isinstance(iterable, list): continue
                            
                        for item in iterable:
                            if not isinstance(item, dict): continue
                            q_text = str(item.get('question', item.get('instruction', ''))).strip()
                            qid = str(item.get('index', item.get('id', '')))
                            pred = str(item.get('pred', item.get('prediction', item.get('model_output', 'A'))))
                            ans = str(item.get('answer', item.get('standard_answer', '')))
                            correct = False
                            if 'score' in item: correct = float(item['score']) > 0
                            elif 'correct' in item: correct = bool(item['correct'])
                            else: correct = (pred.strip() == ans.strip() or pred.strip() in ans.strip() or ans.strip() in pred.strip())
                            
                            if qid and qid != "None":
                                model_preds[m][qid] = pred
                                model_correct[m][qid] = correct
                            if q_text:
                                model_preds[m][q_text[:50]] = pred
                                model_correct[m][q_text[:50]] = correct
                            loaded_cnt += 1
                    except: pass
        if loaded_cnt > 0: print(f"  ✅ [成功] {m}: 加载了 {loaded_cnt} 条预测记录")
        else: print(f"  ❌ [警告] {m}: 未找到评测记录")
    print("-" * 50)
    return model_preds, model_correct

def main():
    args = parse_args()
    print("Loading GAOKAO datasets...")
    ref_data = load_gaokao_data(args.ref_data_dir, args.subjects, args.models)
    test_data = load_gaokao_data(args.test_data_dir, args.subjects, args.models)
    
    print(f"📊 题库规模: 找到了 {len(ref_data)} 道 2023参考题, {len(test_data)} 道 2024测试题")
    if not test_data:
        print("⚠️ 无法加载 GAOKAO 测试数据，请检查 '--subjects' 拼写是否正确。")
        return

    embedder = SentenceTransformer(args.embed_model)
    ref_embs = embedder.encode([item['question'] for item in ref_data], show_progress_bar=False)
    test_embs = embedder.encode([item['question'] for item in test_data], show_progress_bar=True)
    if len(ref_embs) > 0: ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)

    model_preds, model_correct = load_local_results(args.results_dir, args.models)

    use_same_set = False
    ref_has_results = False
    if ref_data:
        ref_q_text = str(ref_data[0].get('question', '')).strip()[:50]
        ref_has_results = any(ref_q_text in model_correct[m] for m in args.models)

    if not ref_has_results:
        print("\n💡 提示: 无 GAOKAO 2023 集评测记录。自动降级 [使用 2024 集留一法交叉参考]\n")
        ref_data, ref_embs, use_same_set = test_data, test_embs, True

    correct_count, total_models_used, early_exit_count = 0, 0, 0
    sims_matrix = np.dot(test_embs, ref_embs.T)

    for i, item in enumerate(test_data):
        qid = str(item.get('question_id', ''))
        qtext = str(item.get('question', '')).strip()[:50]
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
                ref_qid = str(ref_data[idx].get('question_id', ''))
                ref_qtext = str(ref_data[idx].get('question', '')).strip()[:50]
                if model_correct[m].get(ref_qid, model_correct[m].get(ref_qtext, False)): 
                    score += args.beta * weight
            scores[m] = score
            
        models_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        votes, models_used, final_answer = [], 0, "A"
        for m in models_sorted:
            votes.append(model_preds[m].get(qid, model_preds[m].get(qtext, "A")))
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
            
        ans = str(item['answer']).strip()
        if final_answer.strip() == ans or ans in final_answer.strip(): correct_count += 1
        total_models_used += models_used
        
    print("=" * 60)
    print(f"GAOKAO Entropy Routing Acc: {correct_count / max(1,len(test_data)) * 100:.2f}%")
    print(f"Early Exits: {early_exit_count} / {len(test_data)} | Avg Models: {total_models_used / max(1,len(test_data)):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()