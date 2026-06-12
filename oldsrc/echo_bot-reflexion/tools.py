# tools.py
import time
from duckduckgo_search import DDGS
import urllib.request
import xml.etree.ElementTree as ET
import re
import fitz
import os
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

def get_paper_pages(query: str) -> dict:
    """搜索论文，下载PDF并提取前5页作为图片返回"""
    safe_query = urllib.parse.quote(query.strip())
    # 搜索 ArXiv 获取第一条结果
    url = f"http://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results=1&sortBy=relevance&sortOrder=descending"
    
    output_dir = "temp_papers"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        if not entry:
            return {"text": f"未找到与 '{query}' 相关的论文。", "images": []}
        
        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        abstract = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
        
        # 获取 PDF 下载链接
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_url = link.attrib.get('href')
                break
        if not pdf_url:
            pdf_url = entry.find('atom:id', ns).text.replace('abs', 'pdf')
            
        paper_id = pdf_url.split('/')[-1].replace('.pdf', '')
        pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
        
        print(f"[🔧 工具] 正在下载论文并渲染: {title}")
        req_pdf = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_pdf, timeout=15) as res, open(pdf_path, 'wb') as f:
            f.write(res.read())
            
        # 渲染前5页
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(2.0, 2.0)
        image_paths = []
        for page_num in range(min(5, len(doc))):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(output_dir, f"{paper_id}_p{page_num+1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        doc.close()
        
        # 这里返回的是一个字典，方便 agent.py 提取图片路径
        return {
            "text": f"论文《{title}》提取成功。摘要：{abstract[:500]}...",
            "images": image_paths
        }
    except Exception as e:
        return {"text": f"获取论文图片失败: {e}", "images": []}

def search_arxiv(query: str) -> str:
    """搜索 ArXiv 获取最新的学术论文摘要和链接。"""
    
    # 1. 语法清洗：移除可能破坏 ArXiv API 解析的标点符号（尤其是冒号）
    # 仅保留英文字母、数字、短横线和空格
    safe_query = re.sub(r'[^a-zA-Z0-9\s-]', ' ', query).strip()
    # 2. 将多个连续空格压缩为单个空格
    safe_query = re.sub(r'\s+', ' ', safe_query)
    
    # 3. 使用标准库进行安全的 URL 编码
    encoded_query = urllib.parse.quote(safe_query)
    
    # 4. 关键修改：将 sortBy=submittedDate 改为 sortBy=relevance（相关性优先）
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
            link = entry.find('atom:id', ns).text
            published = entry.find('atom:published', ns).text[:10] 
            
            # 截断过长的摘要以防撑爆 LLM 上下文
            if len(summary) > 1024:
                summary = summary[:1021] + "..."
                
            results.append(f"📅 发布时间: {published}\n📄 标题: {title}\n🔗 链接: {link}\n📝 摘要: {summary}")
            
        return "ArXiv 搜索结果：\n\n" + "\n---\n".join(results)
        
    except Exception as e:
        return f"ArXiv 检索失败: {str(e)}"
    
# 记得将它加入你的 AVAILABLE_TOOLS 字典
AVAILABLE_TOOLS = {
    "websearch": websearch,
    "search_arxiv": search_arxiv,
}