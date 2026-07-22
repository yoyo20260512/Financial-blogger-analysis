"""
下载 DeepSeek-R1-Distill-Qwen-1.5B 模型

使用方式:
    python3 download_model.py                          # 默认 HuggingFace
    python3 download_model.py --mirror hf              # HF 镜像 (hf-mirror.com)
    python3 download_model.py --mirror modelscope      # ModelScope (国内最快)
    HF_ENDPOINT=https://hf-mirror.com python3 download_model.py  # 环境变量方式
"""
import os
import sys
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODELSCOPE_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_DIR = Path(__file__).parent / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"


def download_huggingface(mirror=False):
    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("使用 HuggingFace 镜像: https://hf-mirror.com")

    print(f"从 HuggingFace 下载 {MODEL_NAME} ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("下载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("Tokenizer 下载完成")

    print("下载模型权重 (约 3GB) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype="auto", device_map="auto",
    )
    model.save_pretrained(str(MODEL_DIR), safe_serialization=True)
    print("模型下载完成！")


def download_modelscope():
    print(f"从 ModelScope 下载 {MODELSCOPE_NAME} ...")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("安装 modelscope ...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODELSCOPE_NAME, cache_dir=str(MODEL_DIR), local_dir=str(MODEL_DIR))
    print("模型下载完成！")


def download_model(mirror=None):
    if mirror == "modelscope":
        download_modelscope()
    else:
        download_huggingface(mirror=(mirror == "hf"))

    print(f"\n保存路径: {MODEL_DIR}")
    total_size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    print(f"模型总大小: {total_size / 1024**3:.2f} GB")

def verify_model():
    """验证模型是否能正常加载和推理"""
    print("\n验证模型加载...")
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )

    print(f"  - 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  - 设备: {model.device}")

    prompt = "你好，请介绍一下你自己"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
    )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"\n推理测试:")
    print(f"  输入: {prompt}")
    print(f"  输出: {response}")
    print("\n模型验证通过！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mirror", type=str, default=None,
        choices=["hf", "modelscope"],
        help="下载源: hf (HF镜像) / modelscope (魔搭,国内最快)"
    )
    parser.add_argument("--verify-only", action="store_true", help="仅验证已有模型")
    args = parser.parse_args()

    if args.verify_only:
        verify_model()
    else:
        download_model(mirror=args.mirror)
        print("\n" + "=" * 50)
        verify_model()
