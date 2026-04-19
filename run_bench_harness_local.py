# -*- coding: utf-8 -*-
"""
Bench-Harness: 纯本地多卡常驻、真机串行前向推理工作流 (Native Multi-GPU Workflow)
- 物理隔离：专为 8 卡高显存 GPU 节点设计，将 8 个大模型物理映射至各独立显卡 (cuda:0 ~ cuda:7)
- 彻底断网：零 API 依赖，直接在 Python 进程内完成多智能体跨卡通信。
- 严格串行：针对每道空白考题，Executor(前向推理生草稿) -> Reviewer(读取草稿前向推理纠错)
"""

import os
import json
import glob
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. 核心底座：本地多显卡模型资源管理器 (Local VRAM LLM Manager)
# =====================================================================
class LocalModelRegistry:
    """
    负责将传入的 N 个模型权重加载到系统的多张 GPU 上。
    采用“轮询映射（Round-Robin Allocation）”将模型均匀分配到 cuda:0, cuda:1 ...
    """
    def __init__(self, models: list, model_dir: str):
        self.models_dict = {}
        self.tokenizers_dict = {}
        self.model_names = models
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("[❌ 致命错误] 未检测到可用的 GPU！请检查 CUDA 环境。")
            
        print(f"\n" + "=" * 80)
        print(f"🚀 [GPU 资源池] 检测到 {num_gpus} 张可用显卡。开始物理分配 {len(models)} 个模型...")
        print("=" * 80)
        
        for idx, model_name in enumerate(models):
            # 将第 idx 个模型分配到第 (idx % num_gpus) 张卡上
            device = f"cuda:{idx % num_gpus}"
            model_path = os.path.join(model_dir, model_name)
            
            print(f"[*] 显存装载: [{model_name}]  --->  [{device}]")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[!] 找不到模型权重路径: {model_path}")
            
            # 加载 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # 加载 Model (使用 bfloat16 节省显存且保持精度，极大提升前向推理效率)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map={"": device}, # 核心：强制硬绑定到指定 GPU，实现物理隔离
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval() # 切换到评估模式
            
            self.tokenizers_dict[model_name] = tokenizer
            self.models_dict[model_name] = model
            print(f"    [+] {model_name} 成功驻留 {device}。")
            
        print("=" * 80 + "\n")

    @torch.no_grad()
    def generate(self, model_name: str, prompt: str, system_prompt: str = "") -> str:
        """调用指定显卡上的模型进行真实前向推理 (Forward Pass)"""
        model = self.models_dict[model_name]
        tokenizer = self.tokenizers_dict[model_name]
        device = model.device
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 尝试使用官方 Chat Template，如果模型不支持则回退为普通拼接
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:" if system_prompt else f"User: {prompt}\nAssistant:"
        else:
            text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:" if system_prompt else f"User: {prompt}\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # 纯贪心解码 (Greedy Decoding) 保证学术评测的绝对确定性
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 截取新生成的内容
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response


# =====================================================================
# 2. 工具类：数据提取与加载 (Data Loader)
# =====================================================================
class GaokaoDataLoader:
    @staticmethod
    def extract_ground_truth(ans):
        if isinstance(ans, list):
            return "".join([str(a).strip().upper() for a in ans])
        return str(ans).strip().upper()

    @staticmethod
    def extract_model_answer(text: str):
        """强大的正则工具：从大模型的长篇思维链中提取出选项字母"""
        if not text: return ""
        match = re.search(r"【答案】\s*[:：]?\s*([A-Za-z0-9]+)", text)
        if match: return match.group(1).upper()
        fallback = re.findall(r"\b([A-D])\b", text[-50:])
        if fallback: return fallback[-1].upper()
        return ""

    @staticmethod
    def load_calib_matrix(data_dir: str, model_names: list, subjects_filter: list = None):
        """加载校准集(2023-2024)的历史客观题表现矩阵，用于计算数学正交先验"""
        print(f"\n[*] 正在从 {data_dir} 加载历史表现以计算正交图谱...")
        anchor_model = model_names[0]
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
            
            for m_name in model_names:
                model_file = os.path.join(data_dir, f"{m_name}_{keyword}.json")
                if not os.path.exists(model_file):
                    missing_models.append(m_name)
                    valid_dataset = False
                    continue
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        task_data[m_name] = data.get("example", data.get("examples", data))
                except Exception:
                    valid_dataset = False; break
                    
            if not valid_dataset: 
                for m in missing_models: missing_log[m] = missing_log.get(m, 0) + 1
                continue
            
            base_examples = task_data[anchor_model]
            aligned_items = []
            min_len = min(len(task_data[m]) for m in model_names)
            if min_len == 0: continue
            
            for idx in range(min_len):
                q_text = base_examples[idx].get("question", "").strip()
                if not q_text: continue
                std_ans_str = GaokaoDataLoader.extract_ground_truth(base_examples[idx].get("standard_answer", base_examples[idx].get("answer", [])))
                
                row_corr = []
                for m_name in model_names:
                    m_ans_raw = task_data[m_name][idx].get("model_answer", task_data[m_name][idx].get("answer", []))
                    row_corr.append(1 if std_ans_str and std_ans_str == GaokaoDataLoader.extract_ground_truth(m_ans_raw) else 0)
                    
                aligned_items.append({"question": q_text, "model_corr": row_corr})
                
            if aligned_items:
                datasets[keyword] = {"items": aligned_items}
                total_aligned += len(aligned_items)
                
        print(f"[+] 成功对齐 {total_aligned} 道校准题目。")
        if missing_log:
            print("[⚠️ 警告] 以下模型缺失部分校准集文件，对应学科已被安全跳过：")
            for m, count in sorted(missing_log.items(), key=lambda x: -x[1]): print(f"    - [{m}]: 缺失 {count} 个")
        return datasets

    @staticmethod
    def load_raw_test_questions(data_dir: str, test_dir: str, model_names: list, subjects_filter: list = None):
        """【真机在线特有逻辑】直接加载 Objective_Questions 目录下的原始空白高考题用于推理"""
        obj_dir = os.path.join(test_dir, "Objective_Questions")
        print(f"\n[*] 正在从原始空白题库 {obj_dir} 加载盲测集题目...")
        search_pattern = os.path.join(obj_dir, "*.json")
        files = glob.glob(search_pattern)
        
        datasets = {}
        anchor_model = model_names[0]
        
        for file in files:
            keyword = os.path.basename(file).replace(".json", "")
            if subjects_filter and len(subjects_filter) > 0:
                if not any(sub.lower() in keyword.lower() for sub in subjects_filter): continue
                    
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items = data.get("example", data.get("examples", []))
            if not items and isinstance(data, list): items = data
            if not items: continue
                
            # 默认提示词
            prompt = "请你做一道选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】 A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：\n"
            
            # 尝试从 2010-2022 测试集目录中获取该学科原配的 prompt
            anchor_file = os.path.join(test_dir, f"{anchor_model}_{keyword}.json")
            if os.path.exists(anchor_file):
                try:
                    with open(anchor_file, 'r', encoding='utf-8') as f:
                        anchor_data = json.load(f)
                        if isinstance(anchor_data, dict) and "prompt" in anchor_data:
                            prompt = anchor_data["prompt"]
                except Exception: pass

            datasets[keyword] = {"prompt": prompt, "items": items, "raw_file_path": file}
            
        print(f"[+] 成功加载 {len(datasets)} 个待真实推理的空白学科题库。")
        return datasets


# =====================================================================
# 3. 核心算法：Bench-Harness 路由引擎
# =====================================================================
class BenchHarnessEngine:
    def __init__(self, calib_datasets, model_names, args):
        self.args = args
        self.N = len(model_names)
        self.calib_items = []
        for v in calib_datasets.values():
            self.calib_items.extend(v["items"])
            
        self.M = len(self.calib_items)
        if self.M == 0: raise ValueError("[❌ 失败] 校准集为空，请检查历史 JSON 数据或 --models 参数。")
            
        self.Y_calib = np.array([item["model_corr"] for item in self.calib_items])
        self.K = min(self.args.top_k_fixed, self.M) if self.args.top_k_fixed > 0 else max(5, int(self.M * self.args.top_k_ratio))
            
        # 预计算全局宏观先验
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

        print(f"[*] 正在加载流形记忆向量模型 {self.args.embed_model} ...")
        # 向量模型加载到 cpu，不与大模型抢占宝贵的 GPU 显存
        self.embedder = SentenceTransformer(self.args.embed_model, device='cpu')
        calib_questions = [item["question"] for item in self.calib_items]
        self.V_calib = self.embedder.encode(calib_questions, show_progress_bar=True)

    def route(self, q_new_text: str):
        """实例级动态推举最强执行者与最互补审查者"""
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
# 4. 主程序入口：串行真机工作流 (Main Execution)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Bench-Harness 本地多卡真机串行前向推理工作流")
    
    # 🚨 必须指定大模型存放的绝对或相对目录，例如 ./models 
    parser.add_argument("--model_dir", type=str, default="./models", help="本地大模型存放的根目录")
    
    parser.add_argument("--calib_dir", type=str, default="./GAOKAO-Bench-2023-2024/Data")
    parser.add_argument("--test_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    parser.add_argument("--output_dir", type=str, default="./GAOKAO-Bench-2010-2022/Data")
    
    parser.add_argument("--models", type=str, nargs="+", required=True, help="传入 8 个模型的文件名称")
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
    print(f"{'🚀 Bench-Harness 本地多显卡驻留 真机串行前向推理启动 🚀':^110}")
    print("=" * 120)
    print(f"| Models ({len(args.models)}): {args.models}")
    if args.subjects:
        print(f"| Subjects Filter: {args.subjects}")
    print("-" * 120)

    # 1. 抽取历史数据并初始化数学正交底座
    calib_datasets = GaokaoDataLoader.load_calib_matrix(args.calib_dir, args.models, args.subjects)
    if not calib_datasets: return
    engine = BenchHarnessEngine(calib_datasets, args.models, args)
    
    # 2. ⚡⚡ 核心动作：将你的 8 个模型一次性全部加载到系统 GPU 资源池中！
    model_pool = LocalModelPool(args.models, args.model_dir)
    
    # 3. 载入原始空白测试卷，准备实时推理
    test_datasets = GaokaoDataLoader.load_raw_test_questions(args.calib_dir, args.test_dir, args.models, args.subjects)
    if not test_datasets: return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_total_q = 0
    global_exec_corrects = 0
    global_final_corrects = 0
    global_rescue_counts = 0
    subject_metrics = {}
    
    print(f"\n🔥 模型常驻显存完毕！开始执行 OOD 考题的【跨卡串行推理与纠错】...")
    
    for keyword, data in test_datasets.items():
        items = data["items"]
        base_prompt = data["prompt"]
        raw_file_path = data["raw_file_path"]
        bh_output_examples = []
        
        subject_metrics[keyword] = {"total": 0, "exec_corr": 0, "final_corr": 0, "rescue": 0}
        short_desc = keyword[:30] + ".." if len(keyword) > 32 else keyword
        
        for item in tqdm(items, desc=f"GPU Inferencing [{short_desc:<25}]"):
            q_text = item.get("question", "")
            if not q_text: continue
            
            std_ans_str = GaokaoDataLoader.extract_ground_truth(item.get("standard_answer", item.get("answer", [])))
            global_total_q += 1
            subject_metrics[keyword]["total"] += 1
            
            # 【阶段 0：大脑路由分配】
            exec_idx, rev_idx = engine.route(q_text)
            exec_model = args.models[exec_idx]
            rev_model = args.models[rev_idx]
            
            # -------------------------------------------------------------
            # 【阶段 1：执行者前向推理 (调用被指定的 GPU 进行生成)】
            # -------------------------------------------------------------
            exec_sys_prompt = "你是一个全能的高考解题专家。请详细写出你的思考过程与计算步骤，并务必将最终确定的单个选项字母严格写在【答案】和<eoa>之间。例如：【答案】 A <eoa>。"
            exec_prompt = f"{base_prompt}{q_text}"
            
            # 唤醒 exec_model 所在的 GPU 进行前向推理
            exec_draft = model_pool.generate(exec_model, exec_prompt, exec_sys_prompt)
            exec_ans_str = GaokaoDataLoader.extract_model_answer(exec_draft)
            
            # -------------------------------------------------------------
            # 【阶段 2：审查者靶向复核 (跨卡传输草稿，调用另一张 GPU 前向推理)】
            # -------------------------------------------------------------
            rev_sys_prompt = "你是一位极其严苛、极具批判性的学科审查官。请仔细审查并指出草稿中逻辑的断链，给出绝对正确的选项。"
            rev_prompt = (
                f"这是高考原题目：\n------------------------\n{q_text}\n------------------------\n\n"
                f"这是另一位专家（{exec_model}）生成的初步解题草稿：\n"
                f"=================================\n{exec_draft}\n=================================\n\n"
                f"请你作为最终审查官，一步一步严格验证该草稿。\n"
                f"1. 如果草稿中的分析与计算完全正确，请得出相同的结论。\n"
                f"2. 如果你发现草稿存在事实错误、计算漏洞或逻辑幻觉，请严厉指出，并给出你修正后的严密推演。\n\n"
                f"最后，请务必将你最终敲定的正确选项字母写在【答案】和<eoa>之间，例如：【答案】 B <eoa>。"
            )
            
            # 唤醒 rev_model 所在的 GPU 前向推理
            rev_critique = model_pool.generate(rev_model, rev_prompt, rev_sys_prompt)
            final_ans_str = GaokaoDataLoader.extract_model_answer(rev_critique)
            
            # -------------------------------------------------------------
            # 【阶段 3：在线仲裁决断】
            # -------------------------------------------------------------
            if not final_ans_str: 
                final_ans_str = exec_ans_str  # 容错：若审查者未输出明确答案则退回初稿
                
            # -------------------------------------------------------------
            # 【阶段 4：评分与对齐记录】
            # -------------------------------------------------------------
            exec_corr = 1 if std_ans_str and std_ans_str == exec_ans_str else 0
            final_corr = 1 if std_ans_str and std_ans_str == final_ans_str else 0
            
            global_exec_corrects += exec_corr
            global_final_corrects += final_corr
            subject_metrics[keyword]["exec_corr"] += exec_corr
            subject_metrics[keyword]["final_corr"] += final_corr
            
            if exec_corr == 0 and final_corr == 1:
                global_rescue_counts += 1
                subject_metrics[keyword]["rescue"] += 1
                
            ex_out = item.copy()
            ex_out["model_answer"] = [final_ans_str] if final_ans_str else []
            # 记录这极其珍贵的跨智能体对话！
            ex_out["model_output"] = (
                f"【Bench-Harness 真机跨卡协作与辩论轨迹】\n\n"
                f"===== [第一幕：主执行者 (卡 {exec_idx % torch.cuda.device_count()}) {exec_model} 独立草稿] =====\n"
                f"{exec_draft}\n\n"
                f"===== [第二幕：正交审查者 (卡 {rev_idx % torch.cuda.device_count()}) {rev_model} 批判与纠错] =====\n"
                f"{rev_critique}\n\n"
                f"=== [系统最终输出的共识答案: {final_ans_str}] ==="
            )
            bh_output_examples.append(ex_out)
            
        # 写回克隆 JSON，保留原有 metadata
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)
            
        save_data = p_data.copy() if isinstance(p_data, dict) else {}
        save_data["model_name"] = "Bench-Harness-NativeGPU"
        
        if "example" in save_data: save_data["example"] = bh_output_examples
        elif "examples" in save_data: save_data["examples"] = bh_output_examples
        else: save_data["example"] = bh_output_examples
            
        save_path = os.path.join(args.output_dir, f"Bench-Harness-NativeGPU_{keyword}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            
    # =====================================================================
    # 5. 打印真机测试报告矩阵
    # =====================================================================
    print("\n\n" + "=" * 115)
    print(f"{'📊 8卡真机串行推理 学科细粒度性能报告 (Real Multi-GPU Workflow Metrics) 📊':^110}")
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
        print("\n🏆 [宏观真机 8 卡泛化大盘评估报告 (Global Online Report)] 🏆")
        print(f"真机盲测集总规模: {global_total_q} 题 (读取无污染白卷，进行原生前向推理)")
        print("-" * 75)
        
        exec_acc = global_exec_corrects / global_total_q * 100
        final_acc = global_final_corrects / global_total_q * 100
        
        print(f"  🌟 阶段一：纯执行者单卡前向推理准确率 (独狼初稿)       : {exec_acc:.2f}%")
        print(f"  🚀 阶段二：跨卡传送审查者靶向复核后系统成团准确率   : {final_acc:.2f}%")
        print(f"  🔥 真实系统运行中，审查者成功挽救死局的抢救次数     : {global_rescue_counts} 次")
        
        diff = final_acc - exec_acc
        if diff > 0:
            print(f"  📈 System-2 跨卡前向串行推理为系统带来的净胜率提升  : +{diff:.2f}%")
            
        print("=" * 75)
        print(f"[*] ✅ 所有带有真实【思维链辩论草稿】的 JSON 文件已完美保存至:\n    {args.output_dir}/Bench-Harness-NativeGPU_*.json")
        print(f"[*] 这些文件原生兼容 GAOKAO-Bench，可直接运行 local_bench_*.py 进行官方计分！")

if __name__ == "__main__":
    main()