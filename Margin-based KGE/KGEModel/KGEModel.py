import logging
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
import numpy as np
import math

import sys

sys.path.append('/')
from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()

    def init_emb(self):
        # 定义类的时候可以预留不实现的接口在后续继承的子类中实现
        # 为提醒这个类的子类一定要实现这个接口调用NotImplementError
        raise NotImplementedError

    def score_func(self, head_emb, relation_emb, tail_emb):
        raise NotImplementedError

    def forward(self, triples, negs, mode):
        raise NotImplementedError

    def tri2emb(self, triples, mode="single"):
        """
        Get embedding of triples. 正三元组直接使用single模式，负三元组传来positive和negative分mode来转换为embedding
        """
        if mode == 'single':
            head_emb = torch.index_select(self.entity_embedding, dim=0, index=triples[:, 0]).unsqueeze(1)  # 升维
            relation_emb = torch.index_select(self.relation_embedding, dim=0, index=triples[:, 1]).unsqueeze(1)
            tail_emb = torch.index_select(self.entity_embedding, dim=0, index=triples[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = triples
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            if self.args.model_name == 'RESCAL':
                head_emb = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, 1, -1)
            else:
                head_emb = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation_emb = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail_emb = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = triples  # head是positive samples，tail是negative tail_id
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_emb = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)  # batch_size*1*dimension
            relation_emb = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            if self.args.model_name == 'RESCAL':
                tail_emb = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, 1, -1)
            else:
                tail_emb = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
        return head_emb, relation_emb, tail_emb

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        A single train step. Apply back-propation and return the loss
        """
        # 让模型知道现在正在训练。像dropout、batchnorm层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中
        model.train()
        # 每一轮batch需要设置optimizer.zero_grad，根据backward()的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
        # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad了
        optimizer.zero_grad()
        # 取出正负样本
        # positive_sample:tensor:(1024,3) 这是batch个正确的三元组
        # negative_sample:tensor:(1024,256) 这是针对这batch个正三元组的256个负采样的尾/头实体
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            # torch.cuda.set_device(args.device)
            positive_sample = positive_sample.to(args.device)
            negative_sample = negative_sample.to(args.device)
            subsampling_weight = subsampling_weight.to(args.device)

        # (1024,256) 256个负样本的尾实体，针对每一个尾实体有一个负样本的分数
        # 计算负样本分数
        positive_score = model(positive_sample)
        negative_score = model((positive_sample, negative_sample), mode=mode)

        l = model.loss(positive_score, negative_score, subsampling_weight)

        regularization_log = {}

        l.backward()
        optimizer.step()
        log = {**regularization_log, 'loss': l.item()}
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        """
        Evaluate the model on test or valid datasets
        """

        model.eval()  # 测试模式

        if args.countries:  # 针对论文中的country数据集
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.to(args.device)

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)
            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered设置) MRR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )
            test_dataloader_tail = DataLoader(
                TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.to(args.device)
                            negative_sample = negative_sample.to(args.device)
                            filter_bias = filter_bias.to(args.device)

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        # 选择出来的三元组(head,relation,tail1),其他的(head,relation,tail2)也会分数比较高
                        # 对这些分数减去1，降低他们的排名，使其不会干扰对性能的判断（filtered场景）
                        if score.shape != filter_bias.shape:
                            score = score.reshape(batch_size, filter_bias.shape[1])
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        # 分数越大越好 返回从大到小排序后的值所对应原a的下标，即torch.sort()返回的indices
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]  # 预测头，正确头实体的id
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]  # 预测尾，正确尾实体的id
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()  # 得到正确答案的位次
                            assert ranking.size(0) == 1  # 如果ranking.size(0) == 1，程序正常往下运行

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                # 'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)  # 对log的每一个metric都求平均值

        return metrics