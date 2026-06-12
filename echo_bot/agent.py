import re
import json
import os
import copy
from datetime import datetime
import pytz
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import MODEL_LOCAL_PATH
from tools import AVAILABLE_TOOLS

# ==========================================
# 基础配置与全局模型加载
# ==========================================
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

print(">>> 正在加载大模型，请稍候...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_LOCAL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)
print(">>> 大模型加载完成！")

def get_current_time_prompt() -> str:
    tz = pytz.timezone('Asia/Singapore') 
    return f"当前的系统时间是: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %A')}。"

# ==========================================
# 历史记录管理 (精简版)
# ==========================================
def _get_user_file(chat_id: str) -> str:
    safe_chat_id = "".join(c for c in chat_id if c.isalnum() or c in ('-', '_'))
    return os.path.join(HISTORY_DIR, f"{safe_chat_id}.json")

def load_history(chat_id: str, system_prompt: str) -> list:
    user_file = _get_user_file(chat_id)
    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取 {user_file} 失败: {e}")
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

def save_history(chat_id: str, messages: list):
    MAX_HISTORY = 10 
    # 保留 System Prompt 和最近记录
    if len(messages) > MAX_HISTORY + 1:
        messages = [messages[0]] + messages[-MAX_HISTORY:]
    
    # 清洗图片占位符防 OOM
    safe_messages = copy.deepcopy(messages)
    for msg in safe_messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    item.update({"type": "text", "text": "[用户上传了一张图片]"})
                    item.pop("image", None)

    try:
        with open(_get_user_file(chat_id), "w", encoding="utf-8") as f:
            json.dump(safe_messages, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ 写入历史失败: {e}")

def clear_history(chat_id: str):
    user_file = _get_user_file(chat_id)
    if os.path.exists(user_file):
        os.remove(user_file)
        print(f"🧹 已清空 {chat_id} 的记忆")

# ==========================================
# 核心 Agent 逻辑 (支持图文输出接口)
# ==========================================
def ask_agent(chat_id: str, user_prompt: str, is_temp: bool = False, image_path: str = None) -> tuple[str, list]:
    """
    返回: (文本回复, 输出图片路径列表)
    """
    # 优化后的 tools 描述，防止模型“抄作业”乱搜
    tools = [
        {
            "type": "function",
            "function": {
                "name": "websearch",
                "description": "当用户明确询问实时新闻、当前状况时调用。",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "精准的搜索关键词"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_arxiv",
                "description": "当用户的主动提问中明确包含“论文”、“文献”、“学术”等字眼时调用。",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "要搜索的学术关键词，请翻译为英文。"}}, "required": ["query"]}
            }
        }
    ]

    sys_prompt = f"""你是一个具备强大视觉理解(OCR/全能识别)、联网搜索和长期记忆能力的智能助手。
        {get_current_time_prompt()}
        - 当用户发送图片时，请仔细分析图片内容并给出专业的解答。
        - 当你需要获取百科知识或最新信息时，调用 `websearch`。
        - 当你需要查阅学术论文时，调用 `search_arxiv`。
        
        【严格格式指令】:
        当调用 `search_arxiv` 获取结果后，你必须**直接原样列出**论文的「标题」、极其简短的「摘要提取」和「论文编号」。
        **绝对禁止**：
        1. 禁止自行对多篇论文进行综合提炼或总结。
        2. 禁止输出任何过渡性废话（例如“为您找到以下论文...”、”希望对您有帮助”）。
        保持极致的精简和冷酷。
        """
    
    messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}] if is_temp else load_history(chat_id, sys_prompt)
    
    # 构建当前输入
    user_content = []
    if image_path and os.path.exists(image_path):
        user_content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
    user_content.append({"type": "text", "text": user_prompt or "你好"})
    messages.append({"role": "user", "content": user_content})

    is_reflecting, best_answer = False, ""
    output_images = []
    if image_path and os.path.exists(image_path):
        output_images.append(image_path)

    for step in range(8):
        # 恢复：打印轮次信息
        print(f"\n--- [Agent 第 {step + 1} 轮思考开始] ---", flush=True)
        
        text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        out_text = processor.batch_decode([out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)], skip_special_tokens=True)[0]
        
        # 恢复：打印原始输出
        print(f"🤖 [模型本轮原始输出]:\n{out_text}", flush=True)
        
        tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", out_text, re.DOTALL)

        if tool_match:
            is_reflecting = False
            try:
                tool_data = json.loads(tool_match.group(1))
                action, query = tool_data.get("name"), tool_data.get("arguments", {}).get("query")
                
                print(f"🔧 [模型请求调用工具]: {action}, 参数: {query}", flush=True)
                
                obs = AVAILABLE_TOOLS.get(action, lambda x: "Error: Tool not found.")(query) if action else "Invalid tool."
                messages.extend([
                    {"role": "assistant", "content": [{"type": "text", "text": out_text}]},
                    {"role": "tool", "name": action, "content": [{"type": "text", "text": str(obs)}]}
                ])
            except json.JSONDecodeError:
                messages.extend([{"role": "assistant", "content": [{"type": "text", "text": out_text}]}, {"role": "user", "content": [{"type": "text", "text": "JSON 格式有误。"}]}])
        else:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": out_text}]})
            if not is_reflecting:
                best_answer, is_reflecting = out_text, True
                
                # 恢复：打印反思触发提示
                print("🔍 [后台拦截：触发隐式自我反思...]", flush=True)
                
                messages.append({"role": "user", "content": [{"type": "text", "text": "【自检指令】请检查回答是否完美解决问题，且提取了搜索结果关键信息。如需修正请直接输出最新回答；如完美请仅输出 <perfect/>"}]})
                continue
            else:
                if "<perfect/>" not in out_text:
                    print("💡 [反思修正]: 模型发现了不足，给出了优化后的新答案。", flush=True)
                    best_answer = out_text
                else:
                    print("✅ [反思通过]: 模型对原答案非常满意。", flush=True)
                    
                messages = messages[:-2]
                messages[-1] = {"role": "assistant", "content": [{"type": "text", "text": best_answer}]}
                break 

    if not is_temp:
        save_history(chat_id, messages)

    return best_answer.strip(), output_images