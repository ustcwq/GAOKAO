import sys
import os
import json
import argparse
from vllm import LLM, SamplingParams

# Set path and import bench functions
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)

from bench_function import export_distribute_json, export_union_json

class LocalModelAPI:
    def __init__(self, model_path: str, gpu_memory_utilization: float):
        """
        初始化本地大语言模型 (基于 vLLM)
        :param model_path: 模型权重所在的本地文件夹路径
        :param gpu_memory_utilization: GPU 显存利用率
        """
        print(f"Loading local model from {model_path} with vLLM ...")
        self.model_name = os.path.basename(os.path.normpath(model_path))
        
        # 初始化 vLLM
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # 定义采样参数 (greedy decoding)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )
        print("Model loaded successfully.")

    def __call__(self, prompt: str, question: str) -> str:
        """
        对题目进行推理预测
        """
        input_text = prompt + "\n" + question

        messages = [{"role": "user", "content": input_text}]
        
        # 使用官方建议的 Chat Template，如果模型支持
        if getattr(self.tokenizer, "chat_template", None) is not None:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = input_text

        # vLLM 生成
        outputs = self.llm.generate(
            [formatted_prompt],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        response = outputs[0].outputs[0].text.strip()
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAOKAO-Bench Local Evaluation with vLLM")
    parser.add_argument("--model_path", type=str, default="./models/Qwen3.5-9B", help="Path to the local model (e.g., models/Qwen3.5-9B)")
    parser.add_argument("--prompt_file", type=str, default="./Obj_Prompt.json", help="Path to the prompt file (e.g., ./2023_Obj_Prompt.json or ./2024_Obj_Prompt.json)")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU memory utilization for vLLM (default: 0.9)")
    args = parser.parse_args()

    # 读取题目数据文件
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)['examples']

    model_name = os.path.basename(os.path.normpath(args.model_path))

    # 初始化本地模型 API 类
    model_api = LocalModelAPI(args.model_path, args.gpu_util)
        
    for i in range(len(data)):
        directory = "../Data"  # 数据将保存的相对路径
        os.makedirs(directory, exist_ok=True)

        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        
        print("-" * 50)
        print(f"Model ID: {model_name}")
        print(f"Keyword (Subject): {keyword}")
        print(f"Question Type: {question_type}")

        # 调用评测函数生成每道题的分布的 json 
        export_distribute_json(
            model_api, 
            model_name, 
            directory, 
            keyword, 
            zero_shot_prompt_text, 
            question_type, 
            parallel_num=1, 
        )

        export_union_json(
            directory, 
            model_name, 
            keyword,
            zero_shot_prompt_text,
            question_type
        )
