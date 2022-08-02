from load_data import Data
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse


class Experiment:
    def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=100, batch_size=128, decay_rate=0., cuda = False, device=0,
                 input_dropout=0.3, hidden_dropout=0., hidden_dropout1=0.4, hidden_dropout2=0.5,
                 feature_map_dropout=0., in_channels=1, out_channels=32, filt_h=3, filt_w=3,
                 label_smoothing=0.):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.device = device
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "hidden_dropout1": hidden_dropout1, "hidden_dropout2": hidden_dropout2,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

    def get_data_idxs(self, data):
        # 把三元组中的实体和关系都转换为id
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]])
                     for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        # entity-relation词典，记录该头实体+关系对应的尾实体有哪些（可以用于filter-setting或者负采样）
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        # 拿到1个batch_size个(头,关系)，对应拿到(尾)。由于reverse，就不存在head_batch和tail_batch的问题
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.  # 把该条(头,关系)的所有尾实体在target中标1
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.to(self.device)
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        # print("待验证数据集三元组数(reverse): %d" % len(test_data_idxs))
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))  # 数据集全部三元组的(头,关系)->尾 字典

        for i in range(0, len(test_data_idxs), self.batch_size):    # step = batch_size
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)  # data_batch是三元组
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.to(self.device)
                r_idx = r_idx.to(self.device)
                e2_idx = e2_idx.to(self.device)
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):  # 针对data_batch的每一条三元组
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0  # filter-setting，先把所有存在的尾置0，再将真尾设回预测值，避免其他存在尾的影响
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)  # sort_idxs是预测分值从高到低的实体id

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]  # rank是真尾的sort序号
                ranks.append(rank + 1)  # 实际位次要+1

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('MR:{:.6f}, MRR:{:.6f}'.format(np.mean(ranks), np.mean(1. / np.array(ranks))))
        print('Hits@1:{:.6f}, Hits@3:{:.6f}, Hit@10:{:.6f}'.format(np.mean(hits[0]), np.mean(hits[2]), np.mean(hits[9])))

    def train_and_eval(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}  # entity2id
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}  # relation2id

        train_data_idxs = self.get_data_idxs(d.train_data)  # 把train_triples中的实体和关系都转换成id了，reverse过所以double了
        print("训练集三元组数(reverse): %d" % len(train_data_idxs))

        if model_name.lower() == "hype":
            model = HypE(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)
        elif model_name.lower() == "hyper":
            model = HypER(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)
        elif model_name.lower() == "distmult":
            model = DistMult(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)
        elif model_name.lower() == "conve":
            model = ConvE(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)
        elif model_name.lower() == "complex":
            model = ComplEx(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)
        elif model_name.lower() == "tucker":
            model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs)

        if self.cuda:
            model.to(self.device)
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # opt = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())  # 训练集里的(头,关系)

        print("开始训练...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)  # data_batch是(头,关系), target把尾标1
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.to(self.device)
                    r_idx = r_idx.to(self.device)
                predictions = model.forward(e1_idx, r_idx)  # 大小是[batch_size, num_entity]
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print("Epoch " + str(it) + ", time:" + str("%.2f" % (time.time() - start_train)) +
                  ", loss:" + str("%.8f" % np.mean(losses)))
            model.eval()
            with torch.no_grad():
                # print("验证集结果:", end='')
                # self.evaluate(model, d.valid_data)
                # print("训练集结果:", end='')
                # self.evaluate(model, d.train_data)
                if not it % 2:
                    print("测试集结果: ", end='')
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    print("test time:" + str("%.2f" % (time.time() - start_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="ConvE", help='TuckER, HypER, HypE, ConvE, DistMult or ComplEx')
    # FB15k, FB15k-237, WN18 or WN18RR
    parser.add_argument("--dataset", type=str, default="WN18RR", help="数据集")
    parser.add_argument("--cuda", type=bool, default=True, help="use cuda (GPU)")
    parser.add_argument("--device", type=int, default=2)

    args = parser.parse_args()
    model_name = args.model_name
    print("模型: " + model_name)
    data_dir = "../data/%s/" % args.dataset     # 拼接形成数据集路径
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)  # 消除模型训练的随机性，使结果可复现

    # 获取数据，reverse -> (t, r^-1, h)
    d = Data(data_dir=data_dir, reverse=True)     # valid和train也都读进来了，有txt文件就可以
    experiment = Experiment(model_name, num_iterations=200, batch_size=128, learning_rate=0.005,
                            decay_rate=1.0, ent_vec_dim=200, rel_vec_dim=200, cuda=args.cuda, device=args.device,
                            input_dropout=0.2, hidden_dropout=0.3, hidden_dropout1=0.45, hidden_dropout2=0.5,
                            feature_map_dropout=0.2, in_channels=1, out_channels=32, filt_h=1, filt_w=9,
                            label_smoothing=0.1)
    experiment.train_and_eval()
