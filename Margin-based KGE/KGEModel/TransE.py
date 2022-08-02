import torch
import torch.nn as nn
import torch.nn.functional as F
from KGEModel.KGEModel import KGEModel


class TransE(KGEModel):
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.args = args
        self.entity_embedding = None
        self.relation_embedding = None
        self.init_emb()

    def init_emb(self):
        """
            使用均匀分布初始化实体、关系嵌入，entity_dimension = relation_dimension
        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(torch.Tensor([self.args.margin]), requires_grad=False)  # tensor([12.])
        self.embedding_range = nn.Parameter(torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
                                            requires_grad=False)

        self.entity_embedding = nn.Parameter(torch.zeros(self.args.nentity, self.args.emb_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(self.args.nrelation, self.args.emb_dim))
        # 服从由a到b的均匀分布,也就是U～(-0.026,0.026)
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        if mode == 'head-batch':
            score = head_emb + (relation_emb - tail_emb)
        else:
            score = (head_emb + relation_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=2)    # 矩阵范数，p=1列和范数
        return score

    def forward(self, triples, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def loss(self, pos_score, neg_score, subsampling_weight=None):
        neg_score = F.logsigmoid(-neg_score)  # shape:[bs]
        pos_score = F.logsigmoid(pos_score)  # shape:[bs, 1]
        if self.args.uni_weight:
            positive_sample_loss = - pos_score.mean()
            negative_sample_loss = - neg_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * pos_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * neg_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss
