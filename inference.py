"""
DeepSeek-R1-Distill-Qwen-1.5B 推理测试脚本
用法:
    python3 inference.py                   # 使用基础模型
    python3 inference.py --lora            # 使用微调后的 LoRA 模型
    python3 inference.py --interactive     # 交互式对话
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
LORA_DIR = Path(__file__).parent / "outputs" / "lora_finetuned"


def load_model(lora=False):
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )

    if lora:
        print("加载 LoRA 权重...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(LORA_DIR))

    model.eval()
    print(f"设备: {model.device}")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def interactive(model, tokenizer):
    print("\n进入交互模式 (输入 'quit' 退出, '/clear' 清屏)\n")
    history = []
    while True:
        user_input = input("👤 ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.strip() == "/clear":
            print("\n" * 50)
            continue

        print("🤖 思考中...")
        response = generate(model, tokenizer, user_input)
        print(f"\n🤖 {response}\n")
        history.append((user_input, response))


def single_test(model, tokenizer, prompts):
    for prompt in prompts:
        print(f"\n👤 {prompt}")
        print("🤖 思考中...")
        response = generate(model, tokenizer, prompt)
        print(f"🤖 {response}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="使用 LoRA 微调模型")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    model, tokenizer = load_model(lora=args.lora)

    test_prompts = [
        "你好，请介绍一下你自己",
        "请分析当前宏观经济形势对股市的影响",
        "什么是LPR？对房贷有什么影响？",
        "如何看待近期人民币汇率走势？",
    ]

    if args.interactive:
        interactive(model, tokenizer)
    else:
        single_test(model, tokenizer, test_prompts)
