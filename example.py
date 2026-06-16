import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 执行步骤如下:
    # 1. 定义模型路径 path, 并使用 AutoTokenizer 来加载这个模型对应的 tokenizer。
    # 2. 创建一个 LLM 实例 llm, 传入模型路径和一些配置参数, 比如 enforce_eager=True 来强制使用 eager 模式, tensor_parallel_size=1 来设置张量并行的大小。
    # 3. 定义采样参数 sampling_params, 这里设置 temperature=0.6 来控制生成的随机性, max_tokens=256 来限制生成的最大 token 数量。
    # 4. 定义一个包含多个 prompt 的列表 prompts, 这里的 prompt 是一些中文问题。然后使用 tokenizer.apply_chat_template 来把每个 prompt 包装成一个符合聊天格式的输入, 这个函数会根据模型的要求来添加一些特殊的 token 和格式。
    # 5. 调用 llm.generate(prompts, sampling_params) 来生成文本, 这个函数会返回一个列表 outputs, 每个元素是一个字典,包含生成的文本和一些其他信息。
    # 6. 最后使用一个循环来打印每个 prompt 和对应的生成结果。  
    
    path = os.path.expanduser("~/models/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "你是谁",
        "我刚才问的什么问题？",
        "请重复上个问题的答案"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
