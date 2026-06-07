#!/bin/bash
#SBATCH --job-name=feishu_qwen_bot
#SBATCH --gpus=6000ada:1
#SBATCH --time=24:00:00
#SBATCH --output=res/output/job-%j.out
#SBATCH --error=res/error/job-%j.err

# ================= 2. 环境初始化 =================
source ~/.bashrc

if [ -f /etc/profile ]; then
    source /etc/profile
fi

module load Miniforge3
source activate qwen3

echo "Current Python: $(which python)"
echo "Python Version: $(python --version)"

# ================= 3. 依赖与网络检查 =================
echo "Checking and installing dependencies..."
pip install -q lark-oapi duckduckgo-search

# 测试外网
echo "Testing internet connection from compute node..."
curl -I -s --connect-timeout 5 https://duckduckgo.com > /dev/null
if [ $? -eq 0 ]; then
    echo "Internet connection OK. Web search tools will work."
else
    echo "WARNING: Compute node seems disconnected from the internet! Web search tool may fail."
fi

# ================= 4. 运行飞书大模型机器人 =================
echo "Starting Feishu Agent Bot..."

# 切换到你原本机器人的代码目录
cd echo_bot

# 启动修改后的主程序 
python3 -u bot.py