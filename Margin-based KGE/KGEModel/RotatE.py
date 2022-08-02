import torch
import torch.nn as nn
import torch.nn.functional as F
from KGEModel.KGEModel import KGEModel


class RotatE(KGEModel):
    def __init__(self, args):
        super(RotatE, self).__init__(args)
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
        self.entity_dim = 2 * self.args.emb_dim     # 实体维度较关系乘2
        self.entity_embedding = nn.Parameter(torch.zeros(self.args.nentity, self.entity_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(self.args.nrelation, self.args.emb_dim))
        # 服从由a到b的均匀分布,也就是U～(-0.026,0.026)
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head_emb, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation_emb / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.margin.item() - score.sum(dim=2)
        return score

    def forward(self, triples, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def loss(self, pos_score, neg_score, subsampling_weight=None):
        if self.args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            # 自对抗负采样不对抽样权重应用反向传播,也就是(F.softmax(negative_score * args.adversarial_temperature,dim=1)

            neg_score = (F.softmax(neg_score * self.args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-neg_score)).sum(dim=1)
        else:
            neg_score = F.logsigmoid(-neg_score).mean(dim=1)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        if self.args.uni_weight:
            positive_sample_loss = - pos_score.mean()
            negative_sample_loss = - neg_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * pos_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * neg_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss
