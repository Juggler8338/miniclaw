import os
from pyexpat.errors import messages
import re
import json
# from LlamaFactory.src.llamafactory.webui.components import data
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from duckduckgo_search import DDGS
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import time 

# ==========================================
# 1. 飞书凭证配置 (建议通过环境变量传入)
# ==========================================
APP_ID = os.getenv("FEISHU_APP_ID", "你的APP_ID")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "你的APP_SECRET")

# ==========================================
# 2. 全局加载大模型 (只需加载一次)
# ==========================================
model_local_path = "/projects/czkqwen3"
print(">>> 正在加载大模型，请稍候...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_local_path, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_local_path, trust_remote_code=True)
print(">>> 大模型加载完成！")

# ==========================================
# 3. 定义工具与 Agent 逻辑
# ==========================================
def web_search(query: str) -> str:
    """真实的联网搜索工具，使用 DuckDuckGo"""
    query = query.strip().strip('"').strip("'")
    print(f"[🔧 工具执行] 正在全网搜索关键词: {query}")
    time.sleep(2)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "搜索结果：未找到相关信息。请尝试换一组更简短的关键词重新搜索。"
        snippets = [f"- {res['title']}: {res['body']}" for res in results]
        return "搜索结果：\n" + "\n".join(snippets)
    except Exception as e:
        return f"搜索失败: {str(e)}。请尝试精简关键词再试。"

AVAILABLE_TOOLS = {"websearch": web_search}

def ask_agent(user_prompt: str) -> str:
    """处理用户提问，执行多轮 ReAct 逻辑，返回最终答案"""
    agent_system_prompt = """你是一个具备联网搜索能力的智能问答助手。
你可以使用以下工具：
- websearch: 用于在互联网上搜索最新信息。参数必须是简短的英文或中文关键词。

你必须严格按照以下格式进行思考和行动：
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

from concurrent.futures import ThreadPoolExecutor

# 在全局初始化一个线程池，专门用来跑耗时的大模型 Agent
# max_workers=2 表示允许同时处理2个用户的提问（视你的GPU显存而定）
executor = ThreadPoolExecutor(max_workers=2)

def _async_ask_and_reply(data: P2ImMessageReceiveV1, user_message: str):
    """在子线程中运行的大模型推理与回复函数"""
    try:
        print(f"\n[🤖 Agent 开始思考]: {user_message}")
        agent_response = ask_agent(user_message) # 耗时推理
    except Exception as e:
        agent_response = f"大模型处理时发生错误: {str(e)}"
        print(agent_response)

    # 思考完毕后，直接在子线程里组装并发送回复
    content = json.dumps({"text": agent_response})

    if data.event.message.chat_type == "p2p":
        request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            CreateMessageRequestBody.builder().receive_id(data.event.message.chat_id).msg_type("text").content(content).build()
        ).build()
        response = client.im.v1.message.create(request)
    else:
        request = ReplyMessageRequest.builder().message_id(data.event.message.message_id).request_body(
            ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
        ).build()
        response = client.im.v1.message.reply(request)
        
    if not response.success():
        print(f"发送失败, code: {response.code}, msg: {response.msg}")
# ==========================================
# 4. 飞书消息处理核心
# ==========================================
def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    """主线程收到飞书事件后，迅速分发，不留尾巴"""
    if data.event.message.message_type == "text":
        user_message = json.loads(data.event.message.content)["text"]
        print(f"\n[接收消息主线程]: 收到新消息，投递给线程池。")
        
        # 丢给线程池后，该函数直接结束返回，不再向下执行
        executor.submit(_async_ask_and_reply, data, user_message)
        
    else:
        # 非文本消息处理非常快，可以直接在主线程秒回
        content = json.dumps({"text": "我目前只能理解和处理文本消息哦。"})
        if data.event.message.chat_type == "p2p":
            request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
                CreateMessageRequestBody.builder().receive_id(data.event.message.chat_id).msg_type("text").content(content).build()
            ).build()
            client.im.v1.message.create(request)
        else:
            request = ReplyMessageRequest.builder().message_id(data.event.message.message_id).request_body(
                ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
            ).build()
            client.im.v1.message.reply(request)
# ==========================================
# 5. 启动飞书客户端
# ==========================================
event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(do_p2_im_message_receive_v1).build()
client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()
wsClient = lark.ws.Client(APP_ID, APP_SECRET, event_handler=event_handler, log_level=lark.LogLevel.INFO)

if __name__ == "__main__":
    print(">>> 启动长连接监听飞书事件...")
    wsClient.start()