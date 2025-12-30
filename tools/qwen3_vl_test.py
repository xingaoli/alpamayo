import torch
import os
from PIL import Image
import requests
from transformers import Qwen3VLConfig, Qwen3VLProcessor
from transformers import Qwen3VLForConditionalGeneration
from safetensors import safe_open
from transformers.modeling_utils import init_empty_weights

model_path = "ckpts/Qwen3-VL-8B-Instruct-config"
AR1_model_path = "ckpts/Alpamayo-R1-10B"
config = Qwen3VLConfig.from_pretrained(model_path)
config.text_config.vocab_size = 155697
# base_model = Qwen2Model(configuration)
with init_empty_weights():
    model = Qwen3VLForConditionalGeneration(config)

# 1. 直接加载processor, tokenizer含在这里边了（不通过 AutoTokenizer与AutoProcessor）
processor = Qwen3VLProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

# 2. 直接加载模型权重（不通过 AutoModelForCausalLM 与 from_pretrained，方便后续控制）
# 获取所有safetensors文件
safetensors_files = [f for f in os.listdir(AR1_model_path) if f.endswith('.safetensors')]
# 合并所有state_dict
state_dict = {}
for file in safetensors_files:
    file_path = os.path.join(AR1_model_path, file)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key.replace('vlm.', '')] = f.get_tensor(key)

# 加载到模型
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

# model.save_pretrained("ckpts/Alpamayo-R1-Qwen3-VL-8B")
# processor.save_pretrained("ckpts/Alpamayo-R1-Qwen3-VL-8B")

model = model.to('cuda')

# 按需加载，减少内存占用
# safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]

# for file in safetensors_files:
#     file_path = os.path.join(model_path, file)
#     print(f"Loading {file}...")
    
#     with safe_open(file_path, framework="pt", device="cpu") as f:
#         for key in f.keys():
#             # 只加载模型需要的key
#             if hasattr(model, key.split('.')[0]):  # 简单匹配
#                 tensor = f.get_tensor(key)
                
#                 # 获取模型中的对应参数
#                 model_param = model
#                 for attr in key.split('.'):
#                     model_param = getattr(model_param, attr)
                
#                 # 赋值
#                 model_param.data = tensor

# print("Weight loading completed!")


# 只加载特定层的权重（迁移学习场景）
# layers_to_load = ["layers.0", "layers.1", "embed_tokens"]  # 指定要加载的层

# state_dict = {}
# safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]

# for file in safetensors_files:
#     file_path = os.path.join(model_path, file)
#     with safe_open(file_path, framework="pt", device="cpu") as f:
#         for key in f.keys():
#             # 检查是否在要加载的层中
#             if any(layer in key for layer in layers_to_load):
#                 state_dict[key] = f.get_tensor(key)

# model.load_state_dict(state_dict, strict=False)

# 模型太大时，构建好state_dict使用api分发
# model = load_checkpoint_and_dispatch(
#     model,
#     checkpoint=state_dict_path,  # 或者直接传递 state_dict
#     device_map='auto',
#     no_split_module_classes=["Qwen3VLBlock"]  # 指定哪些层不被分割
# )

# 3. 构造对话输入（Qwen2 使用标准 chat template）

messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "图片上有什么？"},
                ],
            },
        ]

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 使用 processor 内置的 apply_chat_template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. 编码
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

# 5. 生成
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id)


# 6. 解码（去掉输入部分）
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)

