import torch
import torch.nn as nn
import torch.nn.functional as F
from KGEModel.KGEModel import KGEModel


class RESCAL(KGEModel):
    def __init__(self, args):
        super(RESCAL, self).__init__(args)
        self.args = args
        self.entity_embedding = None
        self.relation_embedding = None

        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(torch.Tensor([self.args.margin]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
                                            requires_grad=False)

        self.entity_embedding = nn.Parameter(torch.zeros(self.args.nentity, 1, self.args.emb_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(self.args.nrelation, self.args.emb_dim, self.args.emb_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        tail = tail_emb.permute(0, 1, 3, 2)
        if mode == 'head-batch':
            k = torch.matmul(relation_emb, tail)
            score = torch.matmul(head_emb, k)
        else:
            k = torch.matmul(head_emb, relation_emb)
            score = torch.matmul(k, tail)
        score = score.squeeze(dim=2)
        score = score.squeeze(dim=2)
        return score

    def forward(self, triples, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def loss(self, pos_score, neg_score, subsampling_weight=None):
        neg_score = F.logsigmoid(-neg_score)
        pos_score = F.logsigmoid(pos_score)
        if self.args.uni_weight:
            positive_sample_loss = - pos_score.mean()
            negative_sample_loss = - neg_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * pos_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * neg_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss
