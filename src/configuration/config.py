from pathlib import Path

# 根目录
ROOT_PATH = Path(__file__).parent.parent.parent
# 拼写纠错数据集
SPELL_CHECK_DATA_DIR = ROOT_PATH / 'data' / 'spell_check'
# BERT模型
BERT_MODEL_NAME = ROOT_PATH / 'pretrained' / 'bert-base-chinese'
# 模型参数路径
CHECKPOINT_PATH = ROOT_PATH / 'checkpoint'
# 日志目录
LOGS_DIR = ROOT_PATH / 'logs'

# 超参数
SEQ_LEN = 64
