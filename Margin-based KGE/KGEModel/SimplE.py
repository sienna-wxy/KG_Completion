import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from KGEModel.KGEModel import KGEModel


class SimplE(KGEModel):
    def __init__(self, args):
        super(SimplE, self).__init__(args)
        self.args = args
        self.num_ent = args.nentity
        self.num_rel = args.nrelation
        self.emb_dim = args.emb_dim
        self.device = args.device
        self.num_batch = int(math.ceil(float(args.train_triples) / args.batch_size))

        self.ent_h_embs = nn.Embedding(args.nentity, args.emb_dim)
        self.ent_t_embs = nn.Embedding(args.nentity, args.emb_dim)
        self.rel_embs = nn.Embedding(args.nrelation, args.emb_dim)    # 关系的嵌入维度和实体一致
        self.rel_inv_embs = nn.Embedding(args.nrelation, args.emb_dim)

        if args.cuda:
            self.ent_h_embs.to(args.device)
            self.ent_t_embs.to(args.device)
            self.rel_embs.to(args.device)
            self.rel_inv_embs.to(args.device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)    # 均匀分布
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) + (
                    torch.norm(self.rel_embs.weight, p=2) ** 2) + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2

    def forward(self, triples, mode='single'):
        if mode == 'single':
            heads = triples[:, 0].clone().detach()
            rels = triples[:, 1].clone().detach()
            tails = triples[:, 2].clone().detach()
        elif mode == 'head-batch':
            tail_part, head_part = triples
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            heads = head_part.reshape(batch_size * negative_sample_size).clone().detach()
            rels = np.asarray(tail_part[:, 1]).repeat(negative_sample_size, axis=0)
            rels = torch.tensor(rels)
            tails = np.asarray(tail_part[:, 2]).repeat(negative_sample_size, axis=0)
            tails = torch.tensor(tails)
        elif mode == 'tail-batch':
            head_part, tail_part = triples
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            heads = np.asarray(head_part[:, 0]).repeat(negative_sample_size, axis=0)
            heads = torch.tensor(heads)
            rels = np.asarray(head_part[:, 1]).repeat(negative_sample_size, axis=0)
            rels = torch.tensor(rels)
            tails = tail_part.reshape(batch_size * negative_sample_size).clone().detach()

        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def loss(self, pos_score, neg_score, subsampling_weight=None):
        scores = torch.cat([pos_score, neg_score])
        l = torch.cat([torch.ones(pos_score.shape), -1 * torch.ones(neg_score.shape)])
        loss = torch.sum(F.softplus(-l * scores)) + (self.args.regularization * self.l2_loss() / self.num_batch)
        return loss
