"""
DeepSeek-R1-Distill-Qwen-1.5B LoRA 微调脚本
使用 QLoRA 在 Apple Silicon (MPS) 上高效微调

用法:
    python3 finetune_lora.py                        # 默认演示数据
    python3 finetune_lora.py --data my_data.jsonl   # 自定义数据
    python3 finetune_lora.py --test-only            # 测试微调结果
"""
import os
import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer


MODEL_DIR = Path(__file__).parent / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "lora_finetuned"


def load_data(data_path=None):
    """
    加载训练数据
    格式: jsonl，每行 {"instruction": "...", "output": "..."}

    自定义数据示例:
        echo '{"instruction": "什么是CPI", "output": "CPI是居民消费价格指数..."}' > my_data.jsonl
    """
    if data_path and Path(data_path).exists():
        print(f"加载自定义数据: {data_path}")
        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    print("使用示例训练数据 (3条财经演示样本)")
    return [
        {
            "instruction": "请分析当前宏观经济形势对股市的影响",
            "output": "当前宏观经济处于复苏阶段，货币政策保持宽松，财政政策积极发力。"
                      "对股市而言，流动性充裕支撑估值，但基本面修复仍需验证。"
                      "建议关注顺周期板块和政策受益方向，同时注意海外加息周期带来的情绪扰动。"
        },
        {
            "instruction": "解释什么是LPR以及它对房贷利率的影响",
            "output": "LPR（贷款市场报价利率）是商业银行对其最优质客户执行的贷款利率，"
                      "由MLF利率加点形成。当LPR下调时，新发放房贷利率随之降低；"
                      "对于存量房贷，若选择浮动利率，重定价日后月供也会减少。"
                      "因此LPR是房贷利率的基石，直接影响购房成本。"
        },
        {
            "instruction": "如何看待近期人民币汇率走势",
            "output": "近期人民币汇率受内外因素共同影响：外部方面，美元指数走强带来贬值压力；"
                      "内部方面，国内经济复苏节奏和货币政策取向决定汇率中枢。"
                      "短期来看，央行有充足的工具箱维持汇率基本稳定，不会出现单边大幅贬值。"
                      "中长期看，人民币汇率弹性增强，双向波动成为常态。"
        },
    ]


def format_chat_template(example, tokenizer):
    """将数据格式化为对话模板"""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def setup_model():
    """加载模型并配置 LoRA"""
    print("加载基础模型 (4bit量化)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="mps" if torch.backends.mps.is_available() else "auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, dataset, args):
    """执行微调训练"""
    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=1024,
        dataset_text_field="text",
    )

    print("\n开始微调训练...")
    trainer.train()

    print(f"\n保存微调模型: {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))
    print("微调完成!")


def inference_test(model_path=None):
    """测试微调后的模型"""
    if model_path is None:
        model_path = OUTPUT_DIR

    print("\n测试微调后的模型...\n")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(base_model, str(model_path))

    test_prompts = [
        "请分析当前宏观经济形势对股市的影响",
        "解释什么是LPR以及它对房贷利率的影响",
    ]

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"问: {prompt}")
        print(f"答: {response}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 微调脚本")
    parser.add_argument("--data", type=str, default=None, help="训练数据 (jsonl)")
    parser.add_argument("--test-only", action="store_true", help="仅测试已有模型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    args = parser.parse_args()

    if args.test_only:
        inference_test()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        raw_data = load_data(args.data)
        dataset = Dataset.from_list(raw_data)
        model = setup_model()
        train(model, tokenizer, dataset, args)
        inference_test()
