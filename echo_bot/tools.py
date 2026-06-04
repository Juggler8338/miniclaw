# tools.py
import time
from duckduckgo_search import DDGS
import os
import glob

def search_memory(query: str, chat_id: str = None) -> str:
    """搜索过去的对话记录。"""
    # 假设通过某种全局变量或闭包传入 chat_id，或者工具签名调整为接收 chat_id
    user_dir = f"memory_archives/{chat_id}" if chat_id else "memory_archives"
    if not os.path.exists(user_dir):
        return "没有找到相关的历史记忆记录。"
        
    results = []
    # 遍历该用户的所有历史文件
    for filepath in glob.glob(os.path.join(user_dir, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # 简单的文本匹配，如果记录中包含关键词，则提取该段落
            blocks = content.split("---")
            for block in blocks:
                if query.lower() in block.lower():
                    results.append(block.strip())
                    
    if results:
        # 为了防止返回内容过长撑爆上下文，截取最近的 5 条匹配记录
        return "找到以下历史记录：\n\n" + "\n---\n".join(results[-5:])
    else:
        return f"在历史记忆中没有找到与 '{query}' 相关的内容。"

def websearch(query: str) -> str:
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

# 记得将它加入你的 AVAILABLE_TOOLS 字典
AVAILABLE_TOOLS = {
    "websearch": websearch,
    "search_memory": search_memory 
}