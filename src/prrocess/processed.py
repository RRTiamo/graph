from datasets import load_dataset
from transformers import AutoTokenizer

from configuration import config


def process_data(model, save_path):
    """
    数据预处理
    :param model: 处理模型
    :param save_path: 保存数据路径
    """
    # 注意：这里需要的类型是Dataset，方便后续划分数据集
    data_dict = load_dataset("csv", data_files=str(config.SPELL_CHECK_DATA_DIR / 'raw' / 'data.txt'),
                             delimiter=' ', header=None, column_names=['text', 'label'])['train']
    # print(data_dict)
    # 编码
    tokenizer = AutoTokenizer.from_pretrained(model)

    def map_data(batch):
        output_text = tokenizer(batch['text'], truncation=True, max_length=config.SEQ_LEN, padding='max_length')
        input_ids = output_text['input_ids']
        attention_mask = output_text['attention_mask']

        output_label = tokenizer(batch['label'], truncation=True, max_length=config.SEQ_LEN, padding='max_length')
        label = output_label['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    data_dict = data_dict.map(map_data, batched=True, remove_columns=['text', 'label'])
    # 划分数据集
    data_dict = data_dict.train_test_split(test_size=0.2)
    print(data_dict)
    data_dict['test'], data_dict['valid'] = data_dict['test'].train_test_split(test_size=0.5).values()

    # 保存数据集
    data_dict.save_to_disk(save_path)


if __name__ == '__main__':
    process_data('bert-base-chinese', str(config.SPELL_CHECK_DATA_DIR / 'processed' / 'bert'))
