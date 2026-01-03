import os
import time
from dataclasses import dataclass

import datasets
import torch.cuda
from datasets import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from configuration import config
from model.spell_check_bert import SpellCheckModel


# 数据配置类
@dataclass
class TrainingConfig:
    epoch: int = 10
    train_batch_size: int = 8
    test_batch_size: int = 8
    valid_batch_size: int = 8
    lr: float = 5e-5
    # 混合精度训练
    enable_amp: bool = True
    # 早停
    early_stop_patient: int = 3
    early_stop_count: int = 0
    early_stop_strategy: str = 'loss'
    # 最佳损失
    best_score: float = -float('inf')
    # 保存最优模型路径
    output_dir: str = config.CHECKPOINT_PATH
    logs_dir: str = config.LOGS_DIR
    # 打印日志「损失」步数
    logs_step = 50
    # 保存模型步数
    checkpoint_step = 100
    # 评估模型步数
    evaluate_step = 200


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 valid_dataset,
                 train_config,
                 compute_metrics=None,
                 optimizer=None,
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset
        self.train_config = train_config
        self.compute_metrics = compute_metrics
        self.optimizer = optimizer if optimizer else optim.Adam(lr=self.train_config.lr,
                                                                params=self.model.parameters())
        # 创建不存在的文件夹 exist_ok=True: 如果目标路径已存在，makedirs 不会抛出错误，直接跳过。
        os.makedirs(config.CHECKPOINT_PATH / 'spell_check', exist_ok=True)
        # 全局步数记录
        self.global_step = 1
        # 写入tensorboard
        os.makedirs(train_config.logs_dir / 'spell_check', exist_ok=True)
        self.writer = SummaryWriter(log_dir=train_config.logs_dir / 'spell_check' / time.strftime("%Y-%m-%d-%H-%M-%S"))
        # 是否开启混合精度缩放
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.train_config.enable_amp)

    @staticmethod
    def _get_dataloader(dataset, batch_size):
        dataset.set_format(type='torch')
        return DataLoader(dataset, batch_size, shuffle=True)

    def _train_step(self, batch):
        # 准备数据
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label = batch['label'].to(self.device)
        # 混合精度计算
        with torch.autocast(device_type=self.device, dtype=torch.float16,
                            enabled=self.train_config.enable_amp):
            # 前向传播
            output = self.model(input_ids, attention_mask, label)
            # 损失
            loss = output['loss']
        # 清除梯度
        self.optimizer.zero_grad()
        # 反向传播
        self.scaler.scale(loss).backward()
        # 更新参数，一定是先step在update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def trainer(self):
        # 加载检查点
        self._load_checkpoint()
        self.model.train()
        data_loader = self._get_dataloader(dataset=self.train_dataset,
                                           batch_size=self.train_config.train_batch_size)
        # 训练
        for epoch in range(1, 1 + self.train_config.epoch):
            for batch_id, batch in enumerate(tqdm(data_loader, desc=f'EPOCH：{epoch}')):
                # 计算当前的处于的步骤
                current_step = (epoch - 1) * (len(data_loader)) + batch_id
                # 如果当前的步骤比全局步骤小，则跳过（这里是加载断点后的情况）
                if current_step < self.global_step:
                    continue
                loss = self._train_step(batch)
                # 50步记录一次损失值
                if self.global_step % self.train_config.logs_step == 0:
                    tqdm.write(f'[Epoch | {epoch}，Step | {self.global_step}，Loss | {loss}]')
                    self.writer.add_scalar('Loss', loss, self.global_step)

                # 保存模型
                if self.global_step % self.train_config.checkpoint_step == 0:
                    self._save_checkpoint()
                # 评估模型
                if self.global_step % self.train_config.evaluate_step == 0:
                    metric = self.evaluate(dtype='valid')
                    metric_str = " | ".join([f'{k}：{v:.4f}' for k, v in metric.items()])
                    tqdm.write(f'[Epoch | {epoch}，Step | {self.global_step} ] Valid  {metric_str} ')
                    # 早停
                    if self._should_early_stop(metric):
                        return

                self.global_step += 1

    def evaluate(self, dtype='test'):
        self.model.eval()
        data_loader = self._get_dataloader(dataset=self.test_dataset,
                                           batch_size=self.train_config.test_batch_size)
        if dtype == 'valid':
            data_loader = self._get_dataloader(dataset=self.valid_dataset,
                                               batch_size=self.train_config.valid_batch_size)
        total_loss = 0
        all_logits = []
        all_labels = []
        for batch in tqdm(data_loader, desc=f'{dtype}'):
            outputs = self._evaluate_step(batch)
            # 得到一个batch的损失
            batch_loss = outputs['loss']
            total_loss += batch_loss.item()
            if self.compute_metrics is not None:
                # 将所有的标签和输出保存到临时的列表
                all_logits.extend(outputs['pred'].tolist())
                all_labels.extend(batch['label'].tolist())

        # 构造返回参数
        res = {'loss': total_loss / len(data_loader)}
        # 计算评价指标
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(all_logits, all_labels)
            if 'f1' in metrics.keys():
                res['f1'] = metrics['f1'],
            if 'accuracy' in metrics.keys():
                res['accuracy'] = metrics['accuracy']
        return res

    def _evaluate_step(self, batch):
        # 准备数据
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label = batch['label'].to(self.device)
        # 前向传播
        output = self.model(input_ids, attention_mask, label)
        return output

    def _should_early_stop(self, metric):
        score = metric[self.train_config.early_stop_strategy]
        if self.train_config.early_stop_strategy == 'loss':
            score = -score

        if score > self.train_config.best_score:
            # 模型在变好，更新模型参数
            self.train_config.best_score = score
            torch.save(self.model.state_dict(),
                       self.train_config.output_dir / 'spell_check' / 'bert.pt')
            return False
        else:
            self.train_config.early_stop_count += 1
            if self.train_config.early_stop_count >= self.train_config.early_stop_patient:
                return True
            else:
                return False

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            # 需要保存状态字典，如果直接保存模型，会触发安全拦截
            'optimizer': self.optimizer.state_dict(),
            'early_stop_count': self.train_config.early_stop_count,
            'best_score': self.train_config.best_score,
            'global_step': self.global_step,
            # 混合精度
            'scaler_dict': self.scaler.state_dict()
        }
        torch.save(checkpoint, config.CHECKPOINT_PATH / 'spell_check' / 'checkpoint.pt')

    def _load_checkpoint(self):
        checkpoint_path = config.CHECKPOINT_PATH / 'spell_check' / 'checkpoint.pt'
        if checkpoint_path.exists():
            print('断点存在，加载断点')
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.train_config.early_stop_count = checkpoint['early_stop_count']
            self.train_config.best_score = checkpoint['best_score']
            self.global_step = checkpoint['global_step']
            # 混合精度
            self.scaler.load_state_dict(checkpoint['scaler_dict'])
        else:
            print('断点不存在，从头开始训练')


if __name__ == '__main__':
    model = SpellCheckModel(free_param=True)
    data_dict = datasets.load_from_disk(str(config.SPELL_CHECK_DATA_DIR / 'processed' / 'bert'))
    train_config = TrainingConfig()


    def compute_metric(pred, label):
        # pred [[1,2,3,pad,pad],[1,2,3,pad,pad],[1,2,3,pad,pad],[1,2,3,pad]]
        # label [[1,2,3,pad,pad],[1,2,3,pad,pad],[1,2,3,pad,pad],[1,2,3,pad]]
        tokenizers = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

        # pred = tokenizers.batch_decode(pred, skip_special_tokens=True) 直接跳过这样是错误的
        # 因为这里的pred是模型计算得到的，有可能是他给出的token只是一个普通的值，但skip_special_tokens认为他是特殊的值就删了很不合理
        # 借助mask实现跳过逻辑
        total_count = 0
        acc_count = 0
        preds = tokenizers.batch_decode(pred, skip_special_tokens=True)
        labels = tokenizers.batch_decode(label, skip_special_tokens=True)
        for pred, label in zip(preds, labels):
            if pred == label:
                acc_count += 1
            total_count += 1

        return {'accuracy': acc_count / total_count}


    train = Trainer(model,
                    data_dict['train'].select(range(1000)),
                    data_dict['test'],
                    data_dict['valid'].select(range(1000)),
                    train_config,
                    compute_metric
                    )

    train.trainer()
