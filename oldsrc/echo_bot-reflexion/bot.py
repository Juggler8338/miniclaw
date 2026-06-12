# main.py
import json
import os
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from concurrent.futures import ThreadPoolExecutor

from config import APP_ID, APP_SECRET
from agent import ask_agent, clear_history

executor = ThreadPoolExecutor(max_workers=1)
client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()

# 确保图片临时存储目录存在
TEMP_IMG_DIR = "temp_images"
if not os.path.exists(TEMP_IMG_DIR):
    os.makedirs(TEMP_IMG_DIR)

def download_lark_image(message_id: str, image_key: str) -> str:
    """调用飞书 API 下载图片资源到本地"""
    request = GetMessageResourceRequest.builder() \
        .message_id(message_id) \
        .file_key(image_key) \
        .type("image") \
        .build()
    
    response = client.im.v1.message_resource.get(request)
    
    if response.success():
        # 获取二进制文件数据并写入本地
        file_path = os.path.join(TEMP_IMG_DIR, f"{image_key}.jpg")
        with open(file_path, "wb") as f:
            f.write(response.file.read() if hasattr(response.file, 'read') else response.file)
        return file_path
    else:
        print(f"❌ 图片下载失败, code: {response.code}, msg: {response.msg}")
        return None
    
def _async_ask_and_reply(data: P2ImMessageReceiveV1, user_message: str, image_path: str = None):
    """在子线程中运行的大模型推理与回复函数 (已支持图片与 /t)"""
    chat_id = data.event.message.chat_id
    cleaned_message = user_message.strip() if user_message else ""
    
    # 1. 检查清空会话
    if cleaned_message == "/clear":
        print(f"\n[🧹 收到指令]: 正在清空用户 {chat_id} 的历史会话...")
        try:
            clear_history(chat_id)
            agent_response = "✨ 记忆已成功清空！我们开始全新的对话吧。"
        except Exception as e:
            agent_response = f"清空记忆时发生异常: {str(e)}"
            
    # 2. 检查临时会话 /t
    elif cleaned_message.startswith("/t"):
        actual_prompt = cleaned_message[2:].strip()
        if not actual_prompt and not image_path:
            actual_prompt = "你好"
            
        print(f"\n[👻 Agent 开启临时会话 (无痕模式)]: {actual_prompt}")
        try:
            # 传入 is_temp=True 和 image_path
            agent_response = ask_agent(chat_id, actual_prompt, is_temp=True, image_path=image_path)  
        except Exception as e:
            agent_response = f"处理临时会话时发生错误: {str(e)}"
            
    # 3. 常规会话 (图文/纯文本)
    else:
        try:
            print(f"\n[🤖 Agent 开始思考]: 文本='{cleaned_message}', 图片={image_path}")
            agent_response = ask_agent(chat_id, cleaned_message, is_temp=False, image_path=image_path)  
            
        except Exception as e:
            agent_response = f"大模型处理时发生错误: {str(e)}"
        finally:
            # 无论成功还是异常，都确保图片被清理
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    # === 发送回复 ===
    content = json.dumps({"text": agent_response})
    if data.event.message.chat_type == "p2p":
        request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            CreateMessageRequestBody.builder().receive_id(chat_id).msg_type("text").content(content).build()
        ).build()
        response = client.im.v1.message.create(request)
    else:
        request = ReplyMessageRequest.builder().message_id(data.event.message.message_id).request_body(
            ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
        ).build()
        response = client.im.v1.message.reply(request)

def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    """主线程收到飞书事件后，迅速分发到线程池"""
    msg_type = data.event.message.message_type
    message_id = data.event.message.message_id
    
    if msg_type == "text":
        user_message = json.loads(data.event.message.content)["text"]
        print(f"\n[接收消息主线程]: 收到新纯文本消息，投递给线程池。")
        executor.submit(_async_ask_and_reply, data, user_message, None)
        
    elif msg_type == "image":
        image_key = json.loads(data.event.message.content)["image_key"]
        print(f"\n[接收消息主线程]: 收到新图片消息，开始下载...")
        # 1. 下载图片
        local_image_path = download_lark_image(message_id, image_key)
        
        # 2. 飞书发送纯图片时往往没有文本，我们设定一个默认 Prompt
        default_prompt = "请详细分析这张图片中的内容。如果是文档请提取文字(OCR)，如果是场景或物体请识别出图中的关键目标。"
        executor.submit(_async_ask_and_reply, data, default_prompt, local_image_path)
        
    else:
        content = json.dumps({"text": "我目前只支持文本和图片消息。"})
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

# 注册飞书事件回调并初始化长连接客户端
event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(do_p2_im_message_receive_v1).build()
wsClient = lark.ws.Client(APP_ID, APP_SECRET, event_handler=event_handler, log_level=lark.LogLevel.INFO)

if __name__ == "__main__":
    print(">>> 启动长连接监听飞书事件...")
    wsClient.start()