# agent.py
import re
import json
import os          # 新增：用于判断文件是否存在
import threading   # 新增：用于多线程文件锁
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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

# ==========================================
# [核心新增]：持久化状态管理与文件锁
# ==========================================
STATE_FILE = "state.json"
file_lock = threading.Lock()  # 全局文件锁，防止多线程同时读写 state.json 导致文件损坏

def load_history(chat_id: str, system_prompt: str) -> list:
    """从 state.json 加载某个 chat_id 的历史对话，若不存在则初始化"""
    with file_lock:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if chat_id in data:
                        return data[chat_id]
            except Exception as e:
                print(f"读取 {STATE_FILE} 失败，将重新初始化: {e}")
        
        # 没有任何历史，或者读取失败时，返回带 System Prompt 的初始结构
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

def save_history(chat_id: str, messages: list):
    """将某个 chat_id 的历史对话保存到 state.json"""
    with file_lock:
        data = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"读取 {STATE_FILE} 失败，将覆盖创建: {e}")
        
        data[chat_id] = messages
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"写入 {STATE_FILE} 失败: {e}")

def clear_history(chat_id: str):
    """[需求3]：从 state.json 中彻底删除某个 chat_id 的历史记录"""
    with file_lock:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if chat_id in data:
                    del data[chat_id]  # 删除该用户的记忆
                    with open(STATE_FILE, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"🧹 已成功从持久化文件中清除 {chat_id} 的记忆")
            except Exception as e:
                print(f"清空 {chat_id} 记忆失败: {e}")


# ==========================================
# [核心修改]：ask_agent 增加 chat_id 参数
# ==========================================
def ask_agent(chat_id: str, user_prompt: str) -> str:
    """处理用户提问，通过 chat_id 恢复和保存多轮对话历史"""
    
    # 修改 agent.py 中的 tools 列表和 system prompt
    tools = [
        {
            "type": "function",
            "function": {
                "name": "websearch",
                "description": "仅用于检索互联网上的实时、最新信息。",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "用于搜索用户过去的长期对话记忆。当用户提及'以前'、'昨天'、'上次'或者你需要回顾你们的历史交流时使用。",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "用于搜索的关键词"}}, "required": ["query"]}
            }
        }
    ]

    agent_system_prompt = """你是一个具备联网搜索和长期记忆提取能力的智能问答助手。
    - 当涉及最新客观事实时，调用 websearch。
    - 当用户询问你们过去的对话，或需要回忆历史上下文时，调用 search_memory。
    优先使用你的内在知识，得到足够信息后，直接输出最终答案。"""
    
    # [修改点 2]：不再每次都硬编码初始化，而是通过 chat_id 从文件中读取或恢复历史
    messages = load_history(chat_id, agent_system_prompt)
    
    # [修改点 2]：将当前用户输入追加到已恢复的历史记录中
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})

    max_steps = 5  
    for step in range(max_steps):
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        inputs = processor.apply_chat_template(
            messages, 
            tools=tools, 
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

        tool_call_match = re.search(r"<tool_call>\n(.*?)\n</tool_call>", output_text, re.DOTALL)

        if tool_call_match:
            try:
                tool_data = json.loads(tool_call_match.group(1))
                action_name = tool_data.get("name")
                action_input = tool_data.get("arguments", {}).get("query")
                
                print(f"🔧 [模型请求调用工具]: {action_name}, 参数: {action_input}", flush=True)

                if action_name in AVAILABLE_TOOLS:
                    observation = AVAILABLE_TOOLS[action_name](action_input)
                    print(f"👀 [工具返回结果]:\n{observation}", flush=True)
                    
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
            print(f"✅ [成功提取最终答案]: {output_text}", flush=True)
            
            # [修改点 1]：在成功返回最终答案之前，必须将助理的回答也追加入历史，并写回 state.json
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            save_history(chat_id, messages)
            
            return output_text.strip()

    print("❌ [达到最大步数 5，强制退出]", flush=True)
    return "抱歉，我思考了太久，暂时无法给出答案。请重试或换个问法。"