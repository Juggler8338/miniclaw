import os
import re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from duckduckgo_search import DDGS

# ==========================================
# 1. 定义工具 (优化搜索稳定性)
# ==========================================
def web_search(query: str) -> str:
    """真实的联网搜索工具，使用 DuckDuckGo"""
    # 清洗一下关键词，去掉模型可能带上的标点或多余前后缀
    query = query.strip().strip('"').strip("'")
    print(f"\n[🔧 工具执行] 正在全网搜索关键词: {query}")
    try:
        # 使用 ddgs 文本搜索
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return "搜索结果：未找到相关信息。请尝试换一组更简短的关键词重新搜索。"
        
        snippets = [f"- {res['title']}: {res['body']}" for res in results]
        return "搜索结果：\n" + "\n".join(snippets)
    except Exception as e:
        return f"搜索失败: {str(e)}。请尝试精简关键词再试。"

AVAILABLE_TOOLS = {
    "websearch": web_search
}

# ==========================================
# 2. 模型加载
# ==========================================
model_local_path = "/projects/czkqwen3"
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_local_path, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_local_path, trust_remote_code=True)
print("Model loaded successfully.")

# ==========================================
# 3. 优化后的 Agent System Prompt
# ==========================================
agent_system_prompt = """你是一个具备联网搜索能力的智能问答助手。
你可以使用以下工具：
- websearch: 用于在互联网上搜索最新信息。参数必须是简短的英文或中文关键词（例如：'2024 诺贝尔物理学奖'），切勿传入一整句话。

你必须严格按照以下格式进行思考和行动：
Thought: 思考我接下来应该怎么做。是否需要使用工具？
Action: 要使用的工具名称（只能是 websearch，或者为空）
Action Input: 传给工具的简短关键词

注意：当你输出完 'Action Input: [关键词]' 后，必须立刻停止生成，等待系统返回 Observation！绝对不能自己编造 Observation！

如果你已经获得了足够的信息，或者不需要使用工具，请直接输出最终答案：
Thought: 我已经知道答案了。
Final Answer: [你的最终回答]
"""

# ==========================================
# 4. 初始化对话上下文
# ==========================================
user_prompt = "请问2024年诺贝尔物理学奖得主是谁？他们的主要贡献是什么？"
print(f"\n[用户提问]: {user_prompt}")

messages = [
    {"role": "system", "content": [{"type": "text", "text": agent_system_prompt}]},
    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
]

# ==========================================
# 5. Agent 核心执行循环
# ==========================================
max_steps = 5  
print("\n🚀 开启 Agent QA 循环...")

for step in range(max_steps):
    print(f"\n--- 第 {step + 1} 轮思考 ---")
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 生成文本
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # 🌟 【核心修复 1】：如果模型顽固地自己生成了 Observation，强行将其截断，只保留它行动的部分
    if "Observation:" in output_text:
        output_text = output_text.split("Observation:")[0].strip()
    if "观察结果:" in output_text:
        output_text = output_text.split("观察结果:")[0].strip()

    print(f"🤖 模型输出:\n{output_text}")

    # 判断是否完成
    if "Final Answer:" in output_text:
        print("\n✅ 问答任务完成！")
        break

    # 解析模型的 Action 和 Action Input
    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", output_text)
    input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", output_text)

    if action_match and input_match:
        action_name = action_match.group(1).strip()
        action_input = input_match.group(1).strip()

        if action_name in AVAILABLE_TOOLS:
            # 执行本地工具获取真实的观察结果
            observation = AVAILABLE_TOOLS[action_name](action_input)
            print(f"👀 真实观察结果:\n{observation}")
            
            # 将干净的思考过程和系统的观察结果加入上下文
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": f"Observation: {observation}"}]})
        else:
            error_msg = f"Observation: 找不到工具 {action_name}。请只使用 websearch 或者输出 Final Answer。"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": error_msg}]})
    else:
        print("⚠️ 格式错误，提示模型纠正。")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "Observation: 你的输出格式不正确。如果你想搜索，请严格提供 'Action: websearch' 和 'Action Input: [关键词]'。如果已经知道答案，请直接提供 'Final Answer:'。"}]})

else:
    print("\n❌ 达到最大步数限制，强制终止。")