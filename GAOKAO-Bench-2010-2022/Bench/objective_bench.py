import sys
import os
import codecs
import argparse

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)


from vllm import LLM, SamplingParams
from bench_function import export_distribute_json, export_union_json
import os
import json
import time

class LocalModelAPI:
    def __init__(self, model_path: str, gpu_memory_utilization: float):
        self.model_name = os.path.basename(os.path.normpath(model_path))
        print(f"Loading local model from {model_path} with vLLM ...")
        
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )
        print("Model loaded successfully.")

    def __call__(self, prompt: str, question: str) -> str:
        input_text = prompt + "\n" + question
        messages = [{"role": "user", "content": input_text}]
        
        if getattr(self.tokenizer, "chat_template", None) is not None:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = input_text

        outputs = self.llm.generate(
            [formatted_prompt],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        response = outputs[0].outputs[0].text.strip()
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAOKAO-Bench Local Evaluation with vLLM")
    parser.add_argument('--dataset_dir', type=str, default="../Data/Objective_Questions", help="Directory containing the objective questions dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the local model")
    parser.add_argument('--gpu_util', type=float, default=0.8, help="GPU memory utilization for vLLM (e.g., 0.8)")
    args = parser.parse_args()

    # Load the MCQ_prompt.json file
    with open("Obj_Prompt.json", "r") as f:
        data = json.load(f)['examples']

    model_name = os.path.basename(os.path.normpath(args.model_path))
    model_api = LocalModelAPI(args.model_path, args.gpu_util)
        
    for i in range(len(data)):
        directory = args.dataset_dir

        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print("-" * 50)
        print(f"Model ID: {model_name}")
        print(f"Keyword: {keyword}")
        print(f"Question Type: {question_type}")

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
        


