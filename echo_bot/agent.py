# agent.py
from pyexpat.errors import messages
import re
import json
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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
# [修复 3 & 4]：分布式持久化与上下文截断
# ==========================================
HISTORY_DIR = "history"

# 确保历史记录文件夹存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

def _get_user_file(chat_id: str) -> str:
    """获取指定用户的历史记录文件路径"""
    # 替换特殊字符以防路径注入（简单防护）
    safe_chat_id = "".join(c for c in chat_id if c.isalnum() or c in ('-', '_'))
    return os.path.join(HISTORY_DIR, f"{safe_chat_id}.json")

def load_history(chat_id: str, system_prompt: str) -> list:
    """从单独的用户文件中加载历史，若不存在则初始化"""
    user_file = _get_user_file(chat_id)
    
    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取 {user_file} 失败，将重新初始化: {e}")
    
    # 没有任何历史，或者读取失败时，返回带 System Prompt 的初始结构
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

def save_history(chat_id: str, messages: list):
    """保存历史，并限制最大上下文轮数防止 OOM"""
    # [修复 3]：最大保留最近的 10 条消息（5轮对话）
    # 注意：千万不能截掉 index 0 的 system_prompt
    MAX_HISTORY_MESSAGES = 10 
    
    if len(messages) > MAX_HISTORY_MESSAGES + 1:
        # 保留 [0] (System Prompt) 以及最后的 MAX_HISTORY_MESSAGES 条对话
        messages = [messages[0]] + messages[-MAX_HISTORY_MESSAGES:]
        
    user_file = _get_user_file(chat_id)
    try:
        # 直接覆盖写入该用户的专属文件，无需全局锁
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ 写入 {user_file} 失败: {e}")

def clear_history(chat_id: str):
    """彻底删除某个 chat_id 的历史记录文件"""
    user_file = _get_user_file(chat_id)
    if os.path.exists(user_file):
        try:
            os.remove(user_file)
            print(f"🧹 已成功删除文件，清空 {chat_id} 的记忆")
        except Exception as e:
            print(f"❌ 清空 {chat_id} 记忆失败: {e}")

# ==========================================
# [核心修改]：ask_agent 增加 chat_id 参数
# ==========================================
def ask_agent(chat_id: str, user_prompt: str, is_temp: bool = False, image_path: str = None) -> str:
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

    # 稍微优化系统提示词，让它知道自己具备视觉能力
    agent_system_prompt = """你是一个具备强大视觉理解(OCR/全能识别)、联网搜索和长期记忆能力的智能助手。
    - 当用户发送图片时，请仔细分析图片内容（识别物体、提取文档文字、分析图表等），并给出专业的解答。
    - 当用户的提问需要检索最新信息时，调用 `websearch`。
    - 当用户询问历史记忆时，调用 `search_memory`。"""
    
    if is_temp:
        messages = [{"role": "system", "content": [{"type": "text", "text": agent_system_prompt}]}]
    else:
        messages = load_history(chat_id, agent_system_prompt)
    
    # ==========================================
    # [核心修改]：支持图文混合的 Message 构建
    # ==========================================
    user_content = []
    
    # 1. 如果包含图片，将图片对象加入列表
    if image_path and os.path.exists(image_path):
        # Qwen3-VL 支持直接传入本地绝对路径或 file:// URI
        # 此处使用本地路径或 file:// 均可，推荐 url 格式以防 Windows/Linux 路径差异
        user_content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
        
    # 2. 将文本提示词加入列表
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
        
    # 3. 如果既没图片也没文本(防呆兜底)
    if not user_content:
        user_content.append({"type": "text", "text": "你好"})

    # 追加本轮输入
    messages.append({"role": "user", "content": user_content})

    max_steps = 5  
    for step in range(max_steps):
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        # 1. 使用 template 生成包含工具和图片的纯文本 Prompt 字符串 (注意 tokenize=False)
        text = processor.apply_chat_template(
            messages, 
            tools=tools, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. 从 messages 中提取并读取图片文件
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3. 将文本、图片一同传入 processor，在这里才会真正转化出包含 pixel_values 的完整 inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"🤖 [模型本轮原始输出]:\n{output_text}", flush=True)

        # [修复 5]：使用 \s* 匹配可能存在的空格或换行，增加容错率
        tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)

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
            
            # 在返回结果前，先把助手的回答存入 messages
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            
            # [修改点 3]：仅在非临时会话时，才将历史写入磁盘
            if not is_temp:
                save_history(chat_id, messages)
            else:
                print(f"👻 [无痕模式]：本次对话结束，不保存至本地文件。")
            
            return output_text.strip()

    print("❌ [达到最大步数 5，强制退出]", flush=True)
    return "抱歉，我思考了太久，暂时无法给出答案。请重试或换个问法。"