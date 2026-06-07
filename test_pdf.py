import fitz  # PyMuPDF
import urllib.request
import os

def test_download_and_render(pdf_url: str, output_dir: str = "temp_papers"):
    """
    测试下载 ArXiv PDF 并将前 5 页渲染为高清图片。
    """
    # 1. 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建临时文件夹: {output_dir}")

    # 提取论文 ID 作为文件名 (例如从 url 中提取 1706.03762)
    paper_id = pdf_url.split('/')[-1].replace('.pdf', '')
    pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")

    # 2. 下载 PDF
    print(f"⬇️ 正在从 ArXiv 下载 PDF (这可能需要几秒钟)...\n链接: {pdf_url}")
    try:
        # 添加 User-Agent 防止被 ArXiv 拦截
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(pdf_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"✅ 下载完成，保存至: {pdf_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 3. 将前 5 页渲染为图片（截图模式）
    print("\n🖼️ 正在将前 5 页渲染为高清图片...")
    try:
        doc = fitz.open(pdf_path)
        
        # ⚠️ 关键设置：放大渲染矩阵 (Matrix)
        # 默认分辨率太低，大模型可能看不清文字。这里我们将长宽各放大 2 倍，提升图片清晰度
        zoom_x = 2.0  
        zoom_y = 2.0  
        mat = fitz.Matrix(zoom_x, zoom_y)

        # 遍历前 5 页（如果论文不足 5 页则遍历全部）
        for page_num in range(min(5, len(doc))):
            page = doc.load_page(page_num)
            # 将页面渲染为像素图
            pix = page.get_pixmap(matrix=mat)
            
            img_path = os.path.join(output_dir, f"{paper_id}_page_{page_num + 1}.png")
            pix.save(img_path)
            print(f"📸 成功渲染第 {page_num + 1} 页 -> {img_path}")
            
        doc.close()
        print("\n🎉 测试成功！请去 temp_papers 文件夹查看生成的图片。")
        
    except Exception as e:
        print(f"❌ 渲染失败: {e}")

if __name__ == "__main__":
    # 我们用经典的 Transformer 论文 "Attention Is All You Need" 作为测试靶子
    # 这篇论文的 Figure 1 就是非常经典的 Method 架构图
    sample_arxiv_url = "https://arxiv.org/pdf/1706.03762.pdf"
    
    test_download_and_render(sample_arxiv_url)