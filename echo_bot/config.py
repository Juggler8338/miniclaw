# config.py
import os

# 飞书凭证配置
APP_ID = os.getenv("FEISHU_APP_ID", "cli_aa970070c7f89cbc")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "YyHfMnPrO6PPWl2zTTyIKcZS0VkJmwll")

# 大模型本地路径
MODEL_LOCAL_PATH = "/projects/czkqwen3"