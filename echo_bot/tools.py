# tools.py
import time
from duckduckgo_search import DDGS

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

# 注册当前可用工具
AVAILABLE_TOOLS = {"websearch": web_search}