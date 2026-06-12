import os
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from echo_bot.config import MODEL_LOCAL_PATH

print(">>> 1. 加载大模型中...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_LOCAL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)

# 替换为你刚才报错的那张原生飞书图片路径
raw_image_path = "/home/ca0003un/miniclaw/echo_bot/temp_images/img_v3_0212g_eef11f74-6e88-4378-bb3b-4915e4ae598g.jpg"
safe_image_path = "safe_test_image.jpg"

print(f"\n>>> 2. 正在执行图片清洗防御逻辑...")
try:
    img = Image.open(raw_image_path)
    print(f"   [清洗前] 尺寸: {img.size}, 模式: {img.mode}")
    
    # === 核心防御逻辑 ===
    # 1. 防黑底与通道异常
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert('RGB')
        
    # 2. 防显存溢出：限制最大分辨率为 1024
    MAX_DIM = 1024
    if img.width > MAX_DIM or img.height > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
    
    # 3. 防零切片崩溃：防止极端细长/极端矮扁
    PATCH_SIZE = 28
    if img.width < PATCH_SIZE or img.height < PATCH_SIZE:
        scale = max(PATCH_SIZE / img.width, PATCH_SIZE / img.height)
        new_w = max(PATCH_SIZE, int(img.width * scale))
        new_h = max(PATCH_SIZE, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    img.save(safe_image_path, format="JPEG")
    print(f"   [清洗后] 尺寸: {img.size}, 模式: {img.mode} -> 已保存至 {safe_image_path}")
    
except Exception as e:
    print(f"❌ 图片清洗失败: {e}")
    exit()

print("\n>>> 3. 构建对话结构并加载到张量...")
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(safe_image_path)}"}, 
        {"type": "text", "text": "请用一句话简单描述这张图片。"}
    ]}
]

try:
    text = processor.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)
    
    print("\n>>> 4. 🔴 终极测试：进入 GPU 执行前向推理 (model.generate) 🔴")
    # 如果不加清洗逻辑，程序 100% 会死在这一行
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print("\n🎉 完美通过 GPU 考验！没有触发 CUDA 崩溃。")
    print("🤖 模型输出结果：", output_text.strip())

except Exception as e:
    print(f"\n💥 在 GPU 推理环节依然崩溃了: {str(e)}")