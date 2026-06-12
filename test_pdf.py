import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_first_two_figures_ar5iv(arxiv_id):
    # ar5iv 的 HTML 页面地址
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("未找到该论文的 HTML 版本。")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    figures = soup.find_all('figure')
    
    image_urls = []
    # 通常前两个 figure 就是 Motivation 和 Method
    for i, fig in enumerate(figures[:2]): 
        img_tag = fig.find('img')
        if img_tag and 'src' in img_tag.attrs:
            # 拼接完整的图片 URL
            img_url = urllib.parse.urljoin(url, img_tag['src'])
            image_urls.append(img_url)
            
            # 你也可以顺便抓取 caption 来验证是不是 Motivation/Method
            caption = fig.find('figcaption')
            caption_text = caption.text.strip() if caption else "无描述"
            print(f"Figure {i+1} 描述: {caption_text[:50]}...")
            
    return image_urls

# 测试：获取 ResNet 论文 (1512.03385) 的前两张图
images = get_first_two_figures_ar5iv("2210.03629")
print("图片链接:", images)