# agent.py
import re
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 引入配置与工具
from config import MODEL_LOCAL_PATH
from tools import AVAILABLE_TOOLS

# ==========================================
# 全局加载大模型 (保持不变)
# ==========================================
print(">>> 正在加载大模型，请稍候...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_LOCAL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)
print(">>> 大模型加载完成！")

def ask_agent(user_prompt: str) -> str:
    """处理用户提问，使用原生 Function Calling 逻辑，返回最终答案"""
    
    # [核心改动 1]：定义标准 JSON Schema 工具列表
    tools = [
        {
            "type": "function",
            "function": {
                "name": "websearch",
                "description": "仅用于检索互联网上的实时、最新信息。如果无法确定答案，或者涉及最新新闻、价格、版本等时效性信息时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "传给搜索引擎的简短关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # [核心改动 2]：极大精简 System Prompt。不需要再教它怎么换行、怎么输出 Action 了。
    agent_system_prompt = """你是一个具备联网搜索能力的智能问答助手。
优先使用你的内在知识，只有在绝对必要时，才调用提供的 websearch 工具进行搜索。
当你得到了足够的信息后，请直接输出最终答案。"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": agent_system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    max_steps = 5  
    for step in range(max_steps):
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        # [核心改动 3]：将 tools 传入 apply_chat_template
        inputs = processor.apply_chat_template(
            messages, 
            tools=tools, # <--- 这一步让模型知道它拥有工具
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"🤖 [模型本轮原始输出]:\n{output_text}", flush=True)

        # [核心改动 4]：解析模型原生的结构化工具调用标签
        # Qwen 等模型在使用 tools 后，如果要调用工具，通常会输出一段包裹在 XML 标签里的 JSON
        tool_call_match = re.search(r"<tool_call>\n(.*?)\n</tool_call>", output_text, re.DOTALL)

        if tool_call_match:
            # --- 走调用工具的分支 ---
            try:
                tool_data = json.loads(tool_call_match.group(1))
                action_name = tool_data.get("name")
                action_input = tool_data.get("arguments", {}).get("query")
                
                print(f"🔧 [模型请求调用工具]: {action_name}, 参数: {action_input}", flush=True)

                if action_name in AVAILABLE_TOOLS:
                    observation = AVAILABLE_TOOLS[action_name](action_input)
                    print(f"👀 [工具返回结果]:\n{observation}", flush=True)
                    
                    # [核心改动 5]：标准的消息回传格式。将模型的调用请求和工具的返回结果追加到历史中
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                    messages.append({"role": "tool", "name": action_name, "content": [{"type": "text", "text": str(observation)}]})
                else:
                    print(f"⚠️ [找不到请求的工具: {action_name}]", flush=True)
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                    messages.append({"role": "tool", "name": action_name, "content": [{"type": "text", "text": "Error: Tool not found."}]})
                    
            except json.JSONDecodeError:
                print("⚠️ [模型输出的 JSON 格式有误，提示纠正]", flush=True)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                messages.append({"role": "user", "content": [{"type": "text", "text": "你的工具调用格式有误，请确保输出标准的 JSON 结构。"}]})
        else:
            # --- 没有匹配到工具标签，说明模型认为不需要搜索，直接给出了文本答案 ---
            print(f"✅ [成功提取最终答案]: {output_text}", flush=True)
            return output_text.strip()

    print("❌ [达到最大步数 5，强制退出]", flush=True)
    return "抱歉，我思考了太久，暂时无法给出答案。请重试或换个问法。"