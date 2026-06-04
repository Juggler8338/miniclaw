# memory.py
import os
import datetime

MEMORY_DIR = "memory_archives"

def archive_exchange(chat_id: str, user_message: str, agent_response: str):
    """将每一轮对话追加到 Markdown 文件中，作为长期记忆归档"""
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
        
    # 按 chat_id 隔离记忆，避免串号
    user_dir = os.path.join(MEMORY_DIR, str(chat_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        
    # 按天生成文件
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(user_dir, f"{today}.md")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    entry = f"## {timestamp}\n**User**: {user_message}\n**Agent**: {agent_response}\n---\n"
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(entry)