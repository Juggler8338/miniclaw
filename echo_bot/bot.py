import json
import os
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from concurrent.futures import ThreadPoolExecutor

from config import APP_ID, APP_SECRET
from agent import ask_agent, clear_history

from utils import fetch_paper_summary_and_figures
from paper_memory import add_paper_to_list, get_paper_list, delete_paper_from_list

executor = ThreadPoolExecutor(max_workers=1)
client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()

TEMP_IMG_DIR = "temp_images"
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# ==========================================
# 飞书 API 交互工具
# ==========================================
def download_lark_image(message_id: str, image_key: str) -> str:
    """下载用户发送的图片"""
    req = GetMessageResourceRequest.builder().message_id(message_id).file_key(image_key).type("image").build()
    resp = client.im.v1.message_resource.get(req)
    if resp.success():
        path = os.path.join(TEMP_IMG_DIR, f"{image_key}.jpg")
        with open(path, "wb") as f:
            f.write(resp.file.read() if hasattr(resp.file, 'read') else resp.file)
        return path
    return None

def upload_lark_image(file_path: str) -> str:
    """上传本地图片到飞书，获取 image_key 用于发送"""
    with open(file_path, "rb") as f:
        req = CreateImageRequest.builder().request_body(
            CreateImageRequestBody.builder().image_type("message").image(f).build()
        ).build()
        resp = client.im.v1.image.create(req)
        if resp.success():
            return resp.data.image_key
    return None

def send_reply(message_id: str, msg_type: str, content_dict: dict):
    """统一的回复接口 (剔除群聊判断，全部使用串联回复)"""
    req = ReplyMessageRequest.builder().message_id(message_id).request_body(
        ReplyMessageRequestBody.builder().content(json.dumps(content_dict)).msg_type(msg_type).build()
    ).build()
    client.im.v1.message.reply(req)

# ==========================================
# 核心业务处理逻辑
# ==========================================
def _async_ask_and_reply(data: P2ImMessageReceiveV1, user_message: str, image_path: str = None):
    chat_id = data.event.message.chat_id
    msg_id = data.event.message.message_id
    cmd = user_message.strip() if user_message else ""
    
    text_reply = ""
    out_images = []

    try:
        # 1. 业务逻辑处理
        if cmd == "/clear":
            clear_history(chat_id)
            text_reply = "✨ 记忆已成功清空！我们开始全新的对话吧。"
        elif cmd.startswith("/list"):
            if cmd.startswith("/list -d"):
                parts = cmd.split()
                if len(parts) >= 3:
                    text_reply = delete_paper_from_list(chat_id, parts[2].strip())
                else:
                    text_reply = "⚠️ 指令格式错误。请使用 `/list -d <arxiv编号>`"
            elif cmd == "/list":
                text_reply = get_paper_list(chat_id)
            else:
                text_reply = "⚠️ 未知的 list 指令，你是想输入 `/list` 吗？"
                
            send_reply(msg_id, "text", {"text": text_reply})
            return    
        
        # ================= 新增：精准推送论文功能 =================
        elif cmd.startswith("/paper"):
            arxiv_id = cmd[len("/paper"):].strip()
            if not arxiv_id:
                send_reply(msg_id, "text", {"text": "请提供论文的 ArXiv ID，例如: /paper 2310.03744"})
                return
                
            send_reply(msg_id, "text", {"text": f"⏳ 正在前往 ar5iv 拉取论文 {arxiv_id} 的图文，请稍候..."})
            
            # 抓取摘要和图片到本地
            title, abstract, local_images, captions = fetch_paper_summary_and_figures(arxiv_id, TEMP_IMG_DIR)
            add_paper_to_list(chat_id, arxiv_id, title)
            
            # 1. 第一步：单独发送标题和摘要
            abstract_text = f"📄 **{title}**\n\n📝 **Abstract:**\n{abstract}"
            send_reply(msg_id, "text", {"text": abstract_text})
            
            # 2. 第二步：交替发送图片和描述
            # 使用 zip() 将图片路径和描述一一对应
            for i, (img_path, cap) in enumerate(zip(local_images, captions)):
                
                # 先发送图片
                if os.path.exists(img_path):
                    img_key = upload_lark_image(img_path)
                    if img_key:
                        send_reply(msg_id, "image", {"image_key": img_key})
                    os.remove(img_path) # 紧接着清理本地文件
                
                # 再发送该图片对应的描述 (Caption)
                cap_text = f"🖼️ {cap[:512]}..." # 截断防刷屏，可视情况调整长度
                send_reply(msg_id, "text", {"text": cap_text})
                
            return # 执行完指令直接 return，不进入 AI 聊天逻辑
        # ========================================================
        
        elif cmd.startswith("/t"):
            prompt = cmd[2:].strip() or "你好"
            text_reply, out_images = ask_agent(chat_id, prompt, is_temp=True, image_path=image_path)  
        else:
            text_reply, out_images = ask_agent(chat_id, cmd, is_temp=False, image_path=image_path) 
            
        # 2. 发送文本回复
        if text_reply:
            send_reply(msg_id, "text", {"text": text_reply})
            
        # 3. 发送图片回复 (这个时候图片文件还在)
        for img_path in out_images:
            if os.path.exists(img_path):
                img_key = upload_lark_image(img_path)
                if img_key:
                    send_reply(msg_id, "image", {"image_key": img_key})

    except Exception as e:
        send_reply(msg_id, "text", {"text": f"系统异常: {str(e)}"})
        
    finally:
        # 4. 彻底处理完且发送完后，再清理本地临时图片文件
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    msg_type = data.event.message.message_type
    msg_id = data.event.message.message_id
    
    if msg_type == "text":
        text = json.loads(data.event.message.content)["text"]
        executor.submit(_async_ask_and_reply, data, text, None)
        
    elif msg_type == "image":
        img_key = json.loads(data.event.message.content)["image_key"]
        local_path = download_lark_image(msg_id, img_key)
        prompt = "请详细分析这张图片中的内容。如果是文档请提取文字(OCR)，如果是场景或物体请识别出图中的关键目标。"
        executor.submit(_async_ask_and_reply, data, prompt, local_path)
        
    else:
        send_reply(msg_id, "text", {"text": "我目前只支持文本和图片消息。"})

event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(do_p2_im_message_receive_v1).build()
wsClient = lark.ws.Client(APP_ID, APP_SECRET, event_handler=event_handler, log_level=lark.LogLevel.INFO)

if __name__ == "__main__":
    print(">>> 启动长连接监听飞书事件...")
    wsClient.start()