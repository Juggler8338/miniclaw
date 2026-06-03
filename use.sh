#!/bin/bash
#SBATCH --job-name=qwen_agent_qa
#SBATCH --gpus=6000ada:1
#SBATCH --time=1:00:00
#SBATCH --output=res/output/job-%j.out
#SBATCH --error=res/error/job-%j.err

# ================= 环境初始化 =================
source ~/.bashrc

if [ -f /etc/profile ]; then
    source /etc/profile
fi

module load Miniforge3
source activate qwen3

echo "Current Python: $(which python)"
echo "Python Version: $(python --version)"

# ================= 依赖与网络检查 =================
# 确保 Agent 需要的联网搜索库已经安装
pip install -q duckduckgo-search

# 测试计算节点是否能够访问外网 (由于你需要联网搜索，这步很关键)
echo "Testing internet connection from compute node..."
curl -I -s --connect-timeout 5 https://duckduckgo.com > /dev/null
if [ $? -eq 0 ]; then
    echo "Internet connection OK. Web search tools will work."
else
    echo "WARNING: Compute node seems disconnected from the internet! Web search tool may fail."
fi

# ================= 运行模型 =================
echo "Starting Agent QA execution..."
python testmodel.py