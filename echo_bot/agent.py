# agent.py
import re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 引入配置与工具
from config import MODEL_LOCAL_PATH
from tools import AVAILABLE_TOOLS

# ==========================================
# 全局加载大模型 (导入该文件时自动加载一次)
# ==========================================
print(">>> 正在加载大模型，请稍候...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_LOCAL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)
print(">>> 大模型加载完成！")

def ask_agent(user_prompt: str) -> str:
    """处理用户提问，执行多轮 ReAct 逻辑，返回最终答案"""
    agent_system_prompt = """你是一个具备联网搜索能力的智能问答助手。
[核心原则]
1. 优先使用你的内在知识：对于百科常识、数学逻辑、编程代码、翻译或一般性建议，请直接思考并给出 Final Answer，不要使用工具。
2. 仅在必要时搜索：只有当你无法确定答案，或者问题涉及“今天的实时新闻”、“当前的股票价格”、“最近发布的软件版本”等时效性极强的信息时，才使用 websearch。
3. 搜索策略：如果决定搜索，请先在 Thought 中说明理由。

[工具说明]
- websearch: 仅用于检索互联网上的实时、最新信息。

[格式要求]
Thought: 思考我接下来应该怎么做。是否需要使用工具？
Action: 要使用的工具名称（只能是 websearch，或者为空）
Action Input: 传给工具的简短关键词

注意：当你输出完 'Action Input: [关键词]' 后，必须立刻停止生成，等待系统返回 Observation！

如果你已经获得了足够的信息，或者不需要使用工具，请直接输出最终答案：
Thought: 我已经知道答案了。
Final Answer: [你的最终回答]
"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": agent_system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    max_steps = 5  
    for step in range(max_steps):
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 截断自带的 Observation
        if "Observation:" in output_text:
            output_text = output_text.split("Observation:")[0].strip()
        if "观察结果:" in output_text:
            output_text = output_text.split("观察结果:")[0].strip()

        print(f"🤖 [模型本轮原始输出]:\n{output_text}", flush=True)

        # 检查是否得到最终答案
        if "Final Answer:" in output_text:
            final_answer = output_text.split("Final Answer:")[-1].strip()
            print(f"✅ [成功提取最终答案]: {final_answer}", flush=True)
            return final_answer

        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", output_text)
        input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", output_text)

        if action_match and input_match:
            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            print(f"🔧 [准备调用工具]: {action_name}, 参数: {action_input}", flush=True)

            if action_name in AVAILABLE_TOOLS:
                observation = AVAILABLE_TOOLS[action_name](action_input)
                print(f"👀 [工具返回结果]:\n{observation}", flush=True)
                
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                messages.append({"role": "user", "content": [{"type": "text", "text": f"Observation: {observation}"}]})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                messages.append({"role": "user", "content": [{"type": "text", "text": f"Observation: 找不到工具 {action_name}。"}]})
        else:
            print("⚠️ [模型输出格式错误，提示纠正]", flush=True)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": "Observation: 输出格式不正确。请严格按照要求提供 Thought, Action (如果有) 以及 Final Answer。"}]})

    print("❌ [达到最大步数 5，强制退出]", flush=True)
    return "抱歉，我思考了太久，暂时无法给出答案。请重试或换个问法。"