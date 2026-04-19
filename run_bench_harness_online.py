# -*- coding: utf-8 -*-
"""
Bench-Harness: 真实在线前向推理与智能体正交共识工作流 (Online Agent Workflow)
- 支持调用本地部署的 API (如 vLLM) 进行真实的自然语言生成
- 严格串行工作流：执行者先发 (Draft) -> 审查者后审 (Critique & Correction)
- 直接读取原始空白测试题，进行真实前向推理，并生成带有思维链对话轨迹的 JSON 文件
"""

import os
import json
import glob
import re
import time
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================================
# 0. 在线大模型 API 注册表 (LLM API Registry)
# =====================================================================
# 🌟 [极其重要]：请在这里配置你本地启动的模型 API 端口！
# 如果你使用 vLLM 启动模型（例如端口 8000 和 8001），请在这里映射好。
# 键名必须与你命令行传入的 --models 列表名称严格一致。
API_CONFIGS = {
    "Baichuan2-7B-Chat": {"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"},
    "DeepSeek-R1-0528-Qwen3-8B": {"base_url": "http://localhost:8001/v1", "api_key": "EMPTY"},
    "General-Reasoner-7B-preview": {"base_url": "http://localhost:8002/v1", "api_key": "EMPTY"},
    "Llama-3.1-8B-Instruct": {"base_url": "http://localhost:8003/v1", "api_key": "EMPTY"},
    "Qwen3.5-9B": {"base_url": "http://localhost:8004/v1", "api_key": "EMPTY"},
    # TODO: 继续添加你需要参与真机在线测试的模型...
}

class LLMClient:
    """统一管理多个大模型的 HTTP API 请求"""
    def __init__(self, api_configs):
        self.api_configs = api_configs
        self.clients = {}

    def get_client(self, model_name: str):
        if model_name not in self.clients:
            # 默认回退端口 8000
            config = self.api_configs.get(model_name, {"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"})
            self.clients[model_name] = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        return self.clients[model_name]

    def generate(self, model_name: str, prompt: str, system_prompt: str = "", max_retries=3) -> str:
        """真实前向推理调用"""
        client = self.get_client(model_name)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name, # vLLM 默认需要传入启动时的模型名称
                    messages=messages,
                    temperature=0.0,  # 确保学术测试的确定性与可复现性
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"\n[API 警告] {model_name} 请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        return ""


# =====================================================================
# 1. 工具类：数据提取与加载 (Data Loader)
# =====================================================================
class GaokaoDataLoader:
    @staticmethod
    def extract_ground_truth(ans):
        """提取官方标准答案"""
        if isinstance(ans, list):
            return "".join([str(a).strip().upper() for a in ans])
        return str(ans).strip().upper()

    @staticmethod
    def extract_model_answer(text: str):
        """强大的正则工具：从大模型天马行空的思维链中提取出最终选择的字母"""
        if not text: return ""
        # 优先匹配指定格式: 【答案】A<eoa> 或 【答案】: A
        match = re.search(r"【答案】\s*[:：]?\s*([A-Za-z0-9]+)", text)
        if match: return match.group(1).upper()
        
        # 降级匹配：寻找最后出现的独立字母 A-D
        fallback = re.findall(r"\b([A-D])\b", text[-50:])
        if fallback: return fallback[-1].upper()
        return ""

    @staticmethod
    def load_calib_matrix(data_dir: str, models: list, subjects_filter: list = None):
        """加载校准集(2023-2024)的历史数据用于计算正交先验矩阵"""
        print(f"\n[*] 正在从 {data_dir} 加载历史数据用于计算校准宏观先验...")
        anchor_model = models[0]
        search_pattern = os.path.join(data_dir, f"{anchor_model}_*.json")
        anchor_files = glob.glob(search_pattern)
        
        datasets = {}
        total_aligned = 0
        missing_log = {}
        
        for anchor_file in anchor_files:
            if "seperate" in anchor_file or "Subjective" in anchor_file or "Bench-Harness" in anchor_file or "Objective" in anchor_file: 
                continue
                
            keyword = os.path.basename(anchor_file).replace(f"{anchor_model}_", "").replace(".json", "")
            if subjects_filter and len(subjects_filter) > 0:
                if not any(sub.lower() in keyword.lower() for sub in subjects_filter): continue
            
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
                        task_data[model] = data.get("example", data.get("examples", data))
                except Exception:
                    valid_dataset = False; break
                    
            if not valid_dataset: 
                for m in missing_models: missing_log[m] = missing_log.get(m, 0) + 1
                continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            min_len = min(len(task_data[m]) for m in models)
            
            for idx in range(min_len):
                q_text = base_examples[idx].get("question", "").strip()
                if not q_text: continue
                std_ans_str = GaokaoDataLoader.extract_ground_truth(base_examples[idx].get("standard_answer", base_examples[idx].get("answer", [])))
                
                row_corr = []
                for model in models:
                    m_ans_raw = task_data[model][idx].get("model_answer", task_data[model][idx].get("answer", []))
                    m_ans_str = GaokaoDataLoader.extract_ground_truth(m_ans_raw)
                    row_corr.append(1 if std_ans_str and std_ans_str == m_ans_str else 0)
                    
                aligned_items.append({"question": q_text, "model_corr": row_corr})
                
            if aligned_items:
                datasets[keyword] = {"items": aligned_items}
                total_aligned += len(aligned_items)
                
        print(f"[+] 成功对齐 {total_aligned} 道校准题目。")
        if missing_log:
            print("\n[⚠️ 警告] 以下模型缺失部分校准集文件，对应学科已被安全跳过建库：")
            for m, count in sorted(missing_log.items(), key=lambda x: -x[1]): print(f"    - [{m}]: 缺失 {count} 个")
        return datasets

    @staticmethod
    def load_raw_test_questions(data_dir: str, subjects_filter: list = None):
        """【真机在线特有逻辑】直接从 Objective_Questions 目录加载原始的空白高考题！"""
        obj_dir = os.path.join(data_dir, "Objective_Questions")
        print(f"\n[*] 正在从原始空白题库 {obj_dir} 加载待测试题目...")
        search_pattern = os.path.join(obj_dir, "*.json")
        files = glob.glob(search_pattern)
        
        datasets = {}
        for file in files:
            keyword = os.path.basename(file).replace(".json", "")
            if subjects_filter and len(subjects_filter) > 0:
                if not any(sub.lower() in keyword.lower() for sub in subjects_filter): continue
                    
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items = data.get("example", data.get("examples", []))
            if not items and isinstance(data, list): items = data
            if not items: continue
                
            prompt = data.get("prompt", "请你做一道选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：\n")
            datasets[keyword] = {"prompt": prompt, "items": items, "raw_file_path": file}
            
        print(f"[+] 成功加载 {len(datasets)} 个待在线推理的学科题库。")
        return datasets


# =====================================================================
# 3. 核心算法：Bench-Harness 路由引擎
# =====================================================================
class BenchHarnessEngine:
    def __init__(self, calib_datasets, args):
        self.args = args
        self.models = args.models
        self.N = len(args.models)
        self.calib_items = []
        for v in calib_datasets.values():
            self.calib_items.extend(v["items"])
            
        self.M = len(self.calib_items)
        if self.M == 0: raise ValueError("[❌ 失败] 校准集为空。请检查模型名称和数据路径。")
            
        self.Y_calib = np.array([item["model_corr"] for item in self.calib_items])
        self.K = min(self.args.top_k_fixed, self.M) if self.args.top_k_fixed > 0 else max(5, int(self.M * self.args.top_k_ratio))
            
        # 预计算宏观先验
        self.g_acc = np.mean(self.Y_calib, axis=0)
        self.g_crr = np.zeros((self.N, self.N))
        self.g_fdr = np.zeros((self.N, self.N))
        
        for exec_idx in range(self.N):
            err_mask = (self.Y_calib[:, exec_idx] == 0)
            corr_mask = (self.Y_calib[:, exec_idx] == 1)
            num_err, num_corr = np.sum(err_mask), np.sum(corr_mask)
            for rev_idx in range(self.N):
                if exec_idx == rev_idx: continue
                if num_err > 0: self.g_crr[exec_idx, rev_idx] = np.sum(self.Y_calib[err_mask, rev_idx] == 1) / num_err
                if num_corr > 0: self.g_fdr[exec_idx, rev_idx] = np.sum(self.Y_calib[corr_mask, rev_idx] == 0) / num_corr

        print(f"[*] 正在加载向量模型并提取历史记忆特征 (Model: {self.args.embed_model})...")
        self.embedder = SentenceTransformer(self.args.embed_model)
        calib_questions = [item["question"] for item in self.calib_items]
        self.V_calib = self.embedder.encode(calib_questions, show_progress_bar=True)

    def route(self, q_new_text: str):
        """在线检索微观流形并瞬间推选完美搭档"""
        q_new_emb = self.embedder.encode([q_new_text], show_progress_bar=False)[0]
        sims = cosine_similarity(q_new_emb.reshape(1, -1), self.V_calib)[0]
        top_k_indices = np.argsort(sims)[-self.K:]
        
        Y_local = self.Y_calib[top_k_indices]
        exec_idx = int(np.argmax(np.sum(Y_local, axis=0)))
        
        err_mask = (Y_local[:, exec_idx] == 0)
        corr_mask = (Y_local[:, exec_idx] == 1)
        num_err, num_corr = np.sum(err_mask), np.sum(corr_mask)
        
        best_rev_idx, best_obj_score = -1, -float('inf')
        lam, eps = self.args.lambda_smooth, 1e-9 
        
        for j in range(self.N):
            if j == exec_idx: continue
            
            crr_hat = (np.sum(Y_local[err_mask, j] == 1) + lam * self.g_crr[exec_idx, j]) / (num_err + lam + eps)
            fdr_hat = (np.sum(Y_local[corr_mask, j] == 0) + lam * self.g_fdr[exec_idx, j]) / (num_corr + lam + eps)
            acc_hat = (np.sum(Y_local[:, j] == 1) + lam * self.g_acc[j]) / (self.K + lam + eps)
            
            obj_score = self.args.alpha * crr_hat - self.args.beta * fdr_hat + self.args.gamma * acc_hat
            if obj_score > best_obj_score:
                best_obj_score = obj_score
                best_rev_idx = j
                
        if best_rev_idx == -1: best_rev_idx = (exec_idx + 1) % self.N
        return exec_idx, best_rev_idx


# =====================================================================
# 4. 主程序入口 (Main Routine)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Bench-Harness 真实前向推理在线 API 工作流")
    
    parser.add_argument("--calib_dir", type=str, default="./GAOKAO-Bench-2023-2024/Data")
    parser.add_argument("--test_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--output_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    
    parser.add_argument("--top_k_ratio", type=float, default=0.20)
    parser.add_argument("--top_k_fixed", type=int, default=-1)
    parser.add_argument("--lambda_smooth", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 120)
    print(f"{'🚀 Bench-Harness 在线串行真机推理 (Online API Workflow) 启动 🚀':^110}")
    print("=" * 120)

    # 1. 初始化底座引擎与 API 客户端
    calib_datasets = GaokaoDataLoader.load_calib_matrix(args.calib_dir, args.models, args.subjects)
    if not calib_datasets: return
    engine = BenchHarnessEngine(calib_datasets, args)
    llm_client = LLMClient(API_CONFIGS)
    
    # 2. 载入原始空白测试题
    test_datasets = GaokaoDataLoader.load_raw_test_questions(args.test_dir, args.subjects)
    if not test_datasets: return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_total_q = 0
    global_exec_corrects = 0
    global_final_corrects = 0
    global_rescue_counts = 0
    subject_metrics = {}
    
    print(f"\n🏃‍♂️ 正在执行 OOD 考题的串行在线做题与审查纠错...")
    
    for keyword, data in test_datasets.items():
        items = data["items"]
        base_prompt = data["prompt"]
        raw_file_path = data["raw_file_path"]
        bh_output_examples = []
        
        subject_metrics[keyword] = {"total": 0, "exec_corr": 0, "final_corr": 0, "rescue": 0}
        short_desc = keyword[:30] + ".." if len(keyword) > 32 else keyword
        
        for item in tqdm(items, desc=f"Online Evaluating [{short_desc:<30}]"):
            q_text = item.get("question", "")
            if not q_text: continue
            
            std_ans_str = GaokaoDataLoader.extract_ground_truth(item.get("standard_answer", item.get("answer", [])))
            global_total_q += 1
            subject_metrics[keyword]["total"] += 1
            
            # 【阶段 0：路由点将】
            exec_idx, rev_idx = engine.route(q_text)
            exec_model = args.models[exec_idx]
            rev_model = args.models[rev_idx]
            
            # 【阶段 1：执行者前向推理 (生成思维链初稿)】
            exec_sys_prompt = "你是一个全能的高考解题专家。请详细写出你的思考与计算过程，并务必将最终确定的单个选项字母严格写在【答案】和<eoa>之间。例如：【答案】 A <eoa>。"
            exec_prompt = f"{base_prompt}{q_text}"
            
            exec_draft = llm_client.generate(exec_model, exec_prompt, exec_sys_prompt)
            exec_ans_str = GaokaoDataLoader.extract_model_answer(exec_draft)
            
            # 【阶段 2：审查者靶向复核 (前向推理查错)】
            rev_sys_prompt = "你是一位极其严苛、极具批判性的学科审查官。请仔细审查并指出草稿中逻辑的断链，给出绝对正确的选项。"
            rev_prompt = (
                f"这是高考原题目：\n------------------------\n{q_text}\n------------------------\n\n"
                f"这是另一位解题专家（{exec_model}）初步生成的解题草稿：\n"
                f"---------------------------------\n{exec_draft}\n---------------------------------\n\n"
                f"请你作为最终审查官，一步一步严格验证该草稿。\n"
                f"1. 如果草稿中的分析与计算完全正确，请给出相同的答案结论。\n"
                f"2. 如果你发现草稿存在事实错误、计算漏洞或逻辑幻觉，请严厉指出，并给出你修正后的严谨推演。\n\n"
                f"最后，请务必将你最终敲定的正确选项字母写在【答案】和<eoa>之间，例如：【答案】 B <eoa>。"
            )
            
            rev_critique = llm_client.generate(rev_model, rev_prompt, rev_sys_prompt)
            final_ans_str = GaokaoDataLoader.extract_model_answer(rev_critique)
            
            # 【阶段 3：在线仲裁决断】
            if not final_ans_str: 
                final_ans_str = exec_ans_str  # 容错：如果审查者未能输出规范答案，退回执行者答案
                
            # 【阶段 4：评分与统计记录】
            exec_corr = 1 if std_ans_str and std_ans_str == exec_ans_str else 0
            final_corr = 1 if std_ans_str and std_ans_str == final_ans_str else 0
            
            global_exec_corrects += exec_corr
            global_final_corrects += final_corr
            subject_metrics[keyword]["exec_corr"] += exec_corr
            subject_metrics[keyword]["final_corr"] += final_corr
            
            if exec_corr == 0 and final_corr == 1:
                global_rescue_counts += 1
                subject_metrics[keyword]["rescue"] += 1
                
            # 【阶段 5：构建符合 GAOKAO-Bench 的保存格式】
            ex_out = item.copy()
            # 官方测评脚本需要 list 格式的预测结果
            ex_out["model_answer"] = [final_ans_str] if final_ans_str else []
            # 将多智能体的对话轨迹持久化，极其震撼的 Case Study 素材！
            ex_out["model_output"] = (
                f"【Bench-Harness 真机在线协同工作流轨迹】\n\n"
                f"===== [第1步：主执行者 {exec_model} 独立思考草稿] =====\n"
                f"{exec_draft}\n\n"
                f"===== [第2步：正交审查者 {rev_model} 靶向批判与纠错] =====\n"
                f"{rev_critique}\n\n"
                f"=== [系统仲裁决断: {final_ans_str}] ==="
            )
            bh_output_examples.append(ex_out)
            
        # 以克隆原始数据的格式写入 JSON
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)
            
        save_data = p_data.copy() if isinstance(p_data, dict) else {}
        save_data["model_name"] = "Bench-Harness-Online"
        
        # 智能适配 GAOKAO-Bench 数据结构
        if "example" in save_data: save_data["example"] = bh_output_examples
        elif "examples" in save_data: save_data["examples"] = bh_output_examples
        else: save_data["example"] = bh_output_examples
            
        save_path = os.path.join(args.output_dir, f"Bench-Harness-Online_{keyword}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            
    # =====================================================================
    # 5. 打印真机测试报告矩阵
    # =====================================================================
    print("\n\n" + "=" * 115)
    print(f"{'📊 真机串行推理 学科细粒度性能报告 (Online Agent Workflow Metrics) 📊':^110}")
    print("=" * 115)
    
    header = f"| {'Subject (Keyword)':<36} | {'Qs':<4} | {'Exec-Only Acc':<13} | {'Final-BH Acc':<12} | {'Rescue':<6} |"
    print(header)
    print("|" + "-"*38 + "|" + "-"*6 + "|" + "-"*15 + "|" + "-"*14 + "|" + "-"*8 + "|")
    
    for keyword, metrics in sorted(subject_metrics.items()):
        q_count = metrics["total"]
        if q_count == 0: continue
        
        short_kw = keyword.replace("2010-2022_", "").replace("2010-2013_", "").replace("2012-2022_", "")
        if len(short_kw) > 36: short_kw = short_kw[:33] + "..."
        
        exec_acc = (metrics["exec_corr"] / q_count) * 100
        final_acc = (metrics["final_corr"] / q_count) * 100
        rescue_num = metrics["rescue"]
        
        indicator = "🚀" if final_acc > exec_acc else "  "
        print(f"| {short_kw:<36} | {q_count:<4} | {exec_acc:>12.2f}% | {indicator}{final_acc:>9.2f}% | {rescue_num:>6} |")
        
    print("=" * 115)
    
    if global_total_q > 0:
        print("\n🏆 [宏观真机 API 泛化大盘评估报告 (Global Online Report)] 🏆")
        print(f"真机盲测集总规模: {global_total_q} 题 (直接读取未污染的原始题库进行真实生成)")
        print("-" * 75)
        
        exec_acc = global_exec_corrects / global_total_q * 100
        final_acc = global_final_corrects / global_total_q * 100
        
        print(f"  🌟 阶段一：纯执行者 API 前向准确率 (初稿)          : {exec_acc:.2f}%")
        print(f"  🚀 阶段二：经审查者靶向复核 API 后系统成团准确率 : {final_acc:.2f}%")
        print(f"  🔥 真实系统运行中，审查者成功扭转死局的抢救次数  : {global_rescue_counts} 次")
        
        diff = final_acc - exec_acc
        if diff > 0:
            print(f"  📈 System-2 串行交叉验证为系统带来的净胜率提升  : +{diff:.2f}%")
            
        print("=" * 75)
        print(f"[*] ✅ 所有带着真实大模型【辩论思维链】的 JSON 已保存至:\n    {args.output_dir}/Bench-Harness-Online_*.json")
        print(f"[*] 您现在可以直接使用 local_bench_*.py 对其进行评测计分！")

if __name__ == "__main__":
    main()