import numpy as np
import torch
import os
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        # count是字典形式,统计的是(head,relation)和(tail,-relation-1)的相应头尾实体的个数
        self.count = self.count_frequency(triples)
        # 统计(头实体,关系)的尾实体有哪些 && (关系,尾实体)的头实体有哪些
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    # 返回样本个数
    def __len__(self):
        return self.len

    # 返回数据集和标签
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]    # 取出当前的样本(正)
        head, relation, tail = positive_sample

        # 针对这个三元组，计算(head,relation)和(tail,-relation-1)加在一起的次数
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        # 采样权重
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        # 针对这一个正样本生成负样本
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:    # self.negative_sample_size超参，生成多少个负样本
            # np.random.randint返回给定size大小的list，没有写参数high的值，则返回[0,low)的值即[0,nentity)
            # 随机生成negative_sample_size*2个实体的编号，后续要去除真实存在的三元组，double一下保证最后能剩negative_sample_size个
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':    # 替换head
                # numpy.in1d(ar1,ar2,assume_unique=False,invert=False),测试一维数组的每个元素是否也存在于第2个数组中
                # 返回一个长度相同的布尔数组
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], assume_unique=True, invert=True)
            elif self.mode == 'tail-batch':  # 替换tail
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], assume_unique=True, invert=True)
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)

            negative_sample = negative_sample[mask]     # 利用mask把False的head或tail去掉，也就是剔掉真实存在的样本
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]   # 拼接好array取前size个
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):       # 用于将get_item的得到的当前批次的batch_size个ps、ns、sw、m拼成4个tensor
        # data={list:1024}  每个元素是一个tuple:4，tuple的每一个元素分别是正三元组、负采样的的头/尾实体、重采样权重、mode
        positive_sample = torch.stack([_[0] for _ in data], dim=0)   # tensor:(1024,3) 正三元组
        negative_sample = torch.stack([_[1] for _ in data], dim=0)   # tensor:(1024,256) 每一条三元组对应的负采样的头/尾
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)    # tensor:(1024)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec 统计频次用于重采样
        """
        count = {}    # 字典形式，初始值赋start用于平滑？
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build a dictionary of true triples that will  建立正三元组的头/尾字典用于过滤负采样中的真三元组
        be used to filter these true triples for negative sampling
        """
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)    # {(h,r): [t的列表]}
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)    # {(r,t): [h的列表]}

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)    # 去重的所有的正三元组(train/valid/test)
        self.triples = triples    # 测试三元组
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        # tmp把实体集都过一遍，如果有出现过的标-1，否则标0，然后再把这条三元组真正的h/t标为0，这样能找出其他的头or尾
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()     # 保留标签，标-1的说明也是真实存在head/tail
        negative_sample = tmp[:, 1]     # 返回的是实体的id列表

        positive_sample = torch.LongTensor((head, relation, tail))    # positive就返回正常的测试三元组

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        产生一个生成器对象输出dataloader的数据：yield的函数则返回一个可迭代的generator(生成器)对象
        可以使用for循环或者调用next()方法遍历生成器对象来提取结果
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data


def read_triple(dataset, file_path, entity2id, relation2id):
    """
    读取三元组文件，并将实体和关系映射为id.
    """
    triples = []
    if dataset in ['DBpedia50', 'DBpedia500']:
        with open(file_path) as fin:
            for line in fin:
                h, t, r = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    else:
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def get_id(args):
    """
    根据dict文件得到entity2id and relation2id
    """
    # 生成entity to id
    data_path = '../data/' + args.dataset + '/'
    txt_file = data_path + 'entity2id.txt'
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            entity2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        with open(os.path.join(data_path, 'relation2id.txt'), "r") as f:
            relation2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
    else:
        with open(os.path.join(data_path, 'entities.dict')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
        with open(os.path.join(data_path, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    args.nentity = len(entity2id)
    args.nrelation = len(relation2id)
    return entity2id, relation2id
