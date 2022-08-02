import torch
import torch.nn as nn
import torch.nn.functional as F
from KGEModel.KGEModel import KGEModel


class ComplEx(KGEModel):
    def __init__(self, args):
        super(ComplEx, self).__init__(args)
        self.args = args
        self.entity_embedding = None
        self.relation_embedding = None

        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(torch.Tensor([self.args.margin]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
                                            requires_grad=False)
        self.entity_dim = self.args.emb_dim * 2     # 实体维度*2
        self.relation_dim = self.args.emb_dim * 2   # 关系维度*2
        self.entity_embedding = nn.Parameter(torch.zeros(self.args.nentity, self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(self.args.nrelation, self.relation_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        re_head, im_head = torch.chunk(head_emb, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation_emb, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
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

        # Use L3正则 for ComplEx
        regularization = self.args.regularization * (self.entity_embedding.norm(p=3) ** 3 +
                                                     self.relation_embedding.norm(p=3).norm(p=3) ** 3)
        loss = loss + regularization
        return loss
