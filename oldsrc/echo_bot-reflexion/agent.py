import re
import json
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import MODEL_LOCAL_PATH
from tools import AVAILABLE_TOOLS
from datetime import datetime
import pytz # 推荐使用 pytz 处理时区
import copy

def get_current_time_prompt():
    # 强制使用你所在的本地时区
    tz = pytz.timezone('Asia/Singapore') 
    now = datetime.now(tz)
    return f"当前的系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S %A')}。"

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

def clean_messages_for_storage(messages: list) -> list:
    """
    清洗历史记录：遍历所有消息，将图片对象替换为纯文本占位符。
    保留用户的文字提问，丢弃图片路径，彻底根除 FileNotFoundError 和 显存 OOM 隐患。
    """
    cleaned_messages = []
    for msg in messages:
        # 深拷贝以防修改当前仍在内存中运行的原始 messages 对象
        new_msg = copy.deepcopy(msg)
        
        if isinstance(new_msg.get("content"), list):
            for i in range(len(new_msg["content"])):
                item = new_msg["content"][i]
                if item.get("type") == "image":
                    # 将复杂的图片字典替换为简单的纯文本标识
                    new_msg["content"][i] = {
                        "type": "text", 
                        "text": "[用户上传了一张图片]"
                    }
                    
        cleaned_messages.append(new_msg)
        
    return cleaned_messages
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
                "name": "search_arxiv",
                "description": "专用于搜索学术论文、预印本。当用户询问特定领域的最新研究进展、具体论文或前沿技术时调用。",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "搜索关键词，必须翻译为英文。例如 'Joint Image-Time architecture' 或 'flow matching'"
                        }
                    }, 
                    "required": ["query"]
                }
            }
        }
    ]

    # 稍微优化系统提示词，让它知道自己具备视觉能力
    agent_system_prompt = f"""你是一个具备强大视觉理解(OCR/全能识别)、联网搜索和长期记忆能力的智能助手。
    {get_current_time_prompt()}
    - 当用户提问涉及时间（如“今天”、“最近”）时，请参考上述时间。
    - 当用户发送图片时，请仔细分析图片内容（识别物体、提取文档文字、分析图表等），并给出专业的解答。
    - 当用户的提问需要检索最新信息时，调用 `websearch`。
    - 当用户询问学术论文时，调用 `search_arxiv`。"""
    
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

    max_steps = 8  # 适当增加步数，为反思留出空间
    is_reflecting = False # 反思状态标记
    best_answer = ""      # 记录当前最佳答案

    for step in range(max_steps):
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"🤖 [模型本轮原始输出]:\n{output_text}", flush=True)

        tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)

        if tool_call_match:
            # === [有工具调用，维持原样] ===
            is_reflecting = False # 如果它决定调用工具，说明还在解决问题，重置反思状态
            try:
                tool_data = json.loads(tool_call_match.group(1))
                action_name = tool_data.get("name")
                action_input = tool_data.get("arguments", {}).get("query")
                
                print(f"🔧 [模型请求调用工具]: {action_name}, 参数: {action_input}", flush=True)
                if action_name in AVAILABLE_TOOLS:
                    observation = AVAILABLE_TOOLS[action_name](action_input)
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                    messages.append({"role": "tool", "name": action_name, "content": [{"type": "text", "text": str(observation)}]})
                else:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                    messages.append({"role": "tool", "name": action_name, "content": [{"type": "text", "text": "Error: Tool not found."}]})
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                messages.append({"role": "user", "content": [{"type": "text", "text": "你的工具调用格式有误，请确保输出标准的 JSON 结构。"}]})
        
        else:
            # === [模型输出了最终文本，进入反思拦截] ===
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            
            if not is_reflecting:
                # 1. 第一次给出草稿答案，记录为当前最佳答案 (重点：赋值移到了这里)
                best_answer = output_text 
                
                is_reflecting = True
                reflection_prompt = (
                    "【系统内部自检指令】请严格审视你上面的回答。检查：\n"
                    "1. 是否完全、准确地解答了用户的核心问题？\n"
                    "2. 如果调用了搜索工具，是否充分提炼了搜索结果中的关键数据（如论文标题、作者等）？\n"
                    "如果发现不够完美或有错漏，请直接输出改进后的、最完美的完整回答（无需解释原因，直接输出最终文本）。\n"
                    "如果你确信当前回答已经完美且无可挑剔，请务必仅输出标记：<perfect/>"
                )
                print("🔍 [后台拦截：触发隐式自我反思...]", flush=True)
                messages.append({"role": "user", "content": [{"type": "text", "text": reflection_prompt}]})
                continue 
            
            else:
                # 2. 模型已经处于反思状态，检查它的反思结果
                if "<perfect/>" in output_text:
                    print("✅ [反思通过]: 模型对原答案非常满意。", flush=True)
                    # 此时 best_answer 仍保留着第一轮的长篇草稿，不需要覆盖
                    # 关键清理：把我们注入的自检 prompt 和模型的 <perfect/> 回复删掉
                    messages = messages[:-2]
                    break 
                else:
                    print("💡 [反思修正]: 模型发现了不足，给出了优化后的新答案。", flush=True)
                    # 模型输出了新的修正内容，用它覆盖 best_answer (重点：赋值移到了这里)
                    best_answer = output_text
                    
                    # 关键清理：删掉自检 prompt 和这个修正后的回答
                    messages = messages[:-2] 
                    # 将修正后的完美答案替换掉之前的草稿答案，存入历史
                    messages[-1] = {"role": "assistant", "content": [{"type": "text", "text": best_answer}]}
                    break # 轻量级方案，反思修正一次即可，防止陷入死循环

    # === [循环结束，持久化并返回最终结果] ===
    if not is_temp:
        safe_messages = clean_messages_for_storage(messages)
        save_history(chat_id, safe_messages)
    else:
        print(f"👻 [无痕模式]：本次对话结束。")

    return best_answer.strip()
