import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

def fetch_paper_summary_and_figures(arxiv_id: str, save_dir: str) -> tuple[str, str, list, list]:
    """
    根据 ArXiv ID 获取论文标题、摘要，并下载前两张关键图片。
    返回: (标题, 摘要, 本地图片路径列表, 图片描述列表)
    """
    title, abstract = "未知标题", "未找到摘要"
    local_images, captions = [], []
    
    # 1. 使用 ArXiv API 获取精准的 Title 和 Abstract
    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(api_url, timeout=10) as response:
            root = ET.fromstring(response.read())
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            if entry is not None:
                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                abstract = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
    except Exception as e:
        print(f"⚠️ [ArXiv API] 获取摘要失败: {e}")

    # 2. 使用 ar5iv 抓取前两张图片
    html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(html_url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            figures = soup.find_all('figure')
            
            for i, fig in enumerate(figures[:2]): 
                img_tag = fig.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    # 拼接完整 URL 并获取图片数据
                    img_url = urllib.parse.urljoin(html_url, img_tag['src'])
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    
                    # 抓取 Caption
                    caption = fig.find('figcaption')
                    caption_text = caption.text.strip() if caption else f"Figure {i+1}"
                    
                    # 保存到本地 TEMP_IMG_DIR
                    ext = img_url.split('.')[-1][:4] if '.' in img_url else 'png'
                    filename = f"arxiv_{arxiv_id.replace('.', '_')}_fig{i}.{ext}"
                    filepath = os.path.join(save_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                        
                    local_images.append(filepath)
                    captions.append(caption_text)
    except Exception as e:
        print(f"⚠️ [ar5iv] 提取图片失败: {e}")

    return title, abstract, local_images, captions

if __name__ == "__main__":
    arxiv_id = "2210.03629"  # 替换为你想测试的 ArXiv ID
    save_directory = "./temp_images"
    os.makedirs(save_directory, exist_ok=True)
    
    title, abstract, images, captions = fetch_paper_summary_and_figures(arxiv_id, save_directory)
    
    print(f"标题: {title}\n摘要: {abstract}\n图片路径: {images}\n图片描述: {captions}")