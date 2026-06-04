# main.py
import json
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from concurrent.futures import ThreadPoolExecutor

# 引入本项目的其他模块
from config import APP_ID, APP_SECRET
from agent import ask_agent

# 初始化线程池与飞书标准客户端
executor = ThreadPoolExecutor(max_workers=2)
client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()

def _async_ask_and_reply(data: P2ImMessageReceiveV1, user_message: str):
    """在子线程中运行的大模型推理与回复函数"""
    try:
        print(f"\n[🤖 Agent 开始思考]: {user_message}")
        agent_response = ask_agent(user_message)  # 调用 agent.py 里的推理大脑
    except Exception as e:
        agent_response = f"大模型处理时发生错误: {str(e)}"
        print(agent_response)

    # 组装飞书文本消息结构
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

def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    """主线程收到飞书事件后，迅速分发到线程池"""
    if data.event.message.message_type == "text":
        user_message = json.loads(data.event.message.content)["text"]
        print(f"\n[接收消息主线程]: 收到新消息，投递给线程池。")
        
        executor.submit(_async_ask_and_reply, data, user_message)
    else:
        # 非文本消息直接在主线程秒回提示
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

# 注册飞书事件回调并初始化长连接客户端
event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(do_p2_im_message_receive_v1).build()
wsClient = lark.ws.Client(APP_ID, APP_SECRET, event_handler=event_handler, log_level=lark.LogLevel.INFO)

if __name__ == "__main__":
    print(">>> 启动长连接监听飞书事件...")
    wsClient.start()