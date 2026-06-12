# tools.py
import time
from duckduckgo_search import DDGS
import urllib.request
import xml.etree.ElementTree as ET
import re
import urllib.parse

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


def search_arxiv(query: str) -> str:
    """搜索 ArXiv 获取最新的学术论文摘要、链接和论文编号 (ArXiv ID)。"""
    
    safe_query = re.sub(r'[^a-zA-Z0-9\s-]', ' ', query).strip()
    safe_query = re.sub(r'\s+', ' ', safe_query)
    encoded_query = urllib.parse.quote(safe_query)
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=3&sortBy=relevance&sortOrder=descending"
    
    print(f"[🔧 工具执行] 正在 ArXiv 检索论文: {safe_query}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return f"ArXiv 搜索结果：未找到与 '{query}' 相关的论文。请尝试精简关键词。"
            
        results = []
        for entry in entries:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            link = entry.find('atom:id', ns).text # 格式通常为 http://arxiv.org/abs/2310.03744v1
            published = entry.find('atom:published', ns).text[:10] 
            
            # ================= 新增：精准提取 ArXiv ID =================
            arxiv_id = "未知ID"
            id_match = re.search(r'/abs/(\d+\.\d+)', link)
            if id_match:
                arxiv_id = id_match.group(1)
            else:
                arxiv_id = link.split('/')[-1].split('v')[0] # 兜底策略
            # =========================================================

            # if len(summary) > 800:
            #     summary = summary[:797] + "..."
                
            # results.append(f"🆔 【论文编号】: {arxiv_id}\n📅 发布时间: {published}\n📄 标题: {title}\n🔗 链接: {link}\n📝 摘要: {summary}")
            # 将原来 800 字符的截断，改为极其激进的 150 字符
            if len(summary) > 150:
                summary = summary[:147] + "..."
                
            # 简化返回的模版，去掉了发布时间和链接，只保留你最关心的核心信息
            results.append(f" 📄 {title} \n 📅 发布时间: {published} \n📝 {summary} \n 🆔 ID: {arxiv_id}")
        return "ArXiv 搜索结果：\n\n" + "\n---\n".join(results)
        
    except Exception as e:
        return f"ArXiv 检索失败: {str(e)}"
    
AVAILABLE_TOOLS = {
    "websearch": websearch,
    "search_arxiv": search_arxiv,
}