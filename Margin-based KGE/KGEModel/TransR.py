import torch.nn as nn
import torch
import torch.nn.functional as F
from KGEModel.KGEModel import KGEModel


class TransR(KGEModel):
    def __init__(self, args):
        super(TransR, self).__init__(args)
        self.args = args
        self.entity_embedding = None
        self.relation_embedding = None
        self.norm_flag = args.norm_flag
        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(torch.Tensor([self.args.margin]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
                                            requires_grad=False)

        self.entity_embedding = nn.Embedding(self.args.nentity, self.args.emb_dim)
        self.relation_embedding = nn.Embedding(self.args.nrelation, self.args.emb_dim)
        self.transfer_matrix = nn.Embedding(self.args.nrelation, self.args.emb_dim * self.args.emb_dim)
        nn.init.uniform_(tensor=self.entity_embedding.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embedding.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        diag_matrix = torch.eye(self.args.emb_dim)
        diag_matrix = diag_matrix.flatten().repeat(self.args.nrelation, 1)
        self.transfer_matrix.weight.data = diag_matrix

        # nn.init.uniform_(tensor=self.transfer_matrix.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        if self.norm_flag:
            head_emb = F.normalize(head_emb, 2, -1)
            relation_emb = F.normalize(relation_emb, 2, -1)
            tail_emb = F.normalize(tail_emb, 2, -1)
        if mode == "head-batch":
            score = head_emb + (relation_emb - tail_emb)
        else:
            score = (head_emb + relation_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def forward(self, triples, mode="single"):
        """The functions used in the training phase, calculate triple score."""
        if mode == "single":
            head_emb = self.entity_embedding(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.relation_embedding(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.entity_embedding(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
            rel_transfer = self.transfer_matrix(triples[:, 1])  # shape:[bs, dim]

        elif mode == "head-batch":
            tail_part, head_part = triples
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head_emb = torch.index_select(self.entity_embedding.weight.data, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1) # [bs, num_neg, dim]
            relation_emb = self.relation_embedding(tail_part[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.entity_embedding(tail_part[:, 2]).unsqueeze(1)  # [bs, 1, dim]
            rel_transfer = self.transfer_matrix(tail_part[:, 1])  # shape:[bs, dim]

        elif mode == "tail-batch":
            head_part, tail_part = triples  # head是positive samples，tail是negative tail_id
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_emb = self.entity_embedding(head_part[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.relation_embedding(head_part[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = torch.index_select(self.entity_embedding.weight.data, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)# [bs, num_neg, dim]
            rel_transfer = self.transfer_matrix(head_part[:, 1])  # shape:[bs, dim]

        head_emb = self._transfer(head_emb, rel_transfer, mode)
        tail_emb = self._transfer(tail_emb, rel_transfer, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def _transfer(self, emb, rel_transfer, mode):
        rel_transfer = rel_transfer.view(-1, self.args.emb_dim, self.args.emb_dim)
        rel_transfer = rel_transfer.unsqueeze(dim=1)
        emb = emb.unsqueeze(dim=-2)
        emb = torch.matmul(emb, rel_transfer)
        return emb.squeeze(dim=-2)

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
