import torch.cuda
from transformers import AutoTokenizer
from configuration import config
from model.spell_check_bert import SpellCheckModel


class SpellCheckPredictor:
    def __init__(self, model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def predict(self, inputs: list[str] | str):
        # 处理用户输入
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        output_text = self.tokenizer(inputs, truncation=True, max_length=config.SEQ_LEN,
                                     padding='max_length',
                                     return_tensors='pt')
        input_ids = output_text['input_ids'].to(self.device)
        attention_mask = output_text['attention_mask'].to(self.device)
        outputs = self.model(input_ids, attention_mask)
        # 获取预测值
        predictions = outputs['pred']
        # 解码
        results = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        if is_str:
            return results[0]
        else:
            return results


if __name__ == '__main__':
    model = SpellCheckModel()
    # 加载模型
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH / 'spell_check' / 'bert.pt'))
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    scp = SpellCheckPredictor(model, tokenizer)
    res = scp.predict(['我喜混你','日本大藏省一名官员坚称,日本仍忠于全球自由贸易贞经神'])
    print(res)
