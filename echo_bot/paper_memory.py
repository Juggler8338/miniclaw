import json
import os

# 在 agent.py 顶部或此处定义存储目录
PAPER_LIST_DIR = "history/papers"
os.makedirs(PAPER_LIST_DIR, exist_ok=True)

def _get_paper_list_file(chat_id: str) -> str:
    """获取用户的论文列表存储路径"""
    safe_chat_id = "".join(c for c in chat_id if c.isalnum() or c in ('-', '_'))
    return os.path.join(PAPER_LIST_DIR, f"{safe_chat_id}.json")

def add_paper_to_list(chat_id: str, arxiv_id: str, title: str):
    """添加或更新论文到列表（在你的 /paper 逻辑成功获取论文后调用）"""
    file_path = _get_paper_list_file(chat_id)
    papers = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                papers = json.load(f)
        except Exception:
            pass
            
    # 使用 arxiv_id 作为 key，避免重复添加
    papers[arxiv_id] = title
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)

def get_paper_list(chat_id: str) -> str:
    """获取用户的论文列表"""
    file_path = _get_paper_list_file(chat_id)
    if not os.path.exists(file_path):
        return "📂 你的论文记录为空。"
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except Exception:
        return "⚠️ 读取论文列表失败。"
        
    if not papers:
        return "📂 你的论文记录为空。"
        
    res = ["📚 **你查询过的论文列表：**\n"]
    for idx, (aid, title) in enumerate(papers.items(), 1):
        res.append(f"{idx}. **[{aid}]** {title}")
        
    res.append("\n💡 *提示：输入 `/list -d <arxiv编号>` 即可删除特定记录。*")
    return "\n".join(res)

def delete_paper_from_list(chat_id: str, arxiv_id: str) -> str:
    """从列表中删除指定论文"""
    file_path = _get_paper_list_file(chat_id)
    if not os.path.exists(file_path):
        return "📂 你的论文记录为空，无法删除。"
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
            
        if arxiv_id in papers:
            title = papers.pop(arxiv_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, ensure_ascii=False, indent=4)
            return f"✅ 已成功从记录中删除：\n**[{arxiv_id}]** {title}"
        else:
            return f"❌ 未找到编号为 **{arxiv_id}** 的论文记录。"
    except Exception as e:
        return f"⚠️ 删除失败：{str(e)}"