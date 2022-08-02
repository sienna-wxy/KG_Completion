# --do_train --cuda --do_valid --do_test --data_path ../data/FB15k --model RotatE -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv -lr 0.0001 --max_epoch 150000 -save ../models/RotatE_FB15k_0 --test_batch_size 16 -de

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from KGEModel import TransE, TransR, TransH, RESCAL, DistMult, ComplEx, RotatE, SimplE
from dataloader import TrainDataset, BidirectionalOneShotIterator, read_triple, get_id
from utils import override_config, save_model, set_logger, log_metrics


def setup_parser():
    """
    设置模型超参数，注意修改模型保存路径
    """
    parser = argparse.ArgumentParser(description='Knowledge Graph Completion', usage='train.py [<args>] [-h | --help]')

    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
    parser.add_argument('--device', type=str, default="cuda:4", help='use GPU or CPU')
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', default=False, help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, help='国家数据集的区域ID，自动生成')

    datasets = ['FB15k', 'FB15k-237', 'WN18', 'WN18RR', 'YAGO3-10', 'DBpedia50', 'DBpedia500', 'Kinship', 'Nations',
                'UMLS']
    parser.add_argument('--dataset', type=str, default='UMLS', choices=datasets, help='数据集路径')
    parser.add_argument('--model_name', default='TransH', type=str, help='模型')

    # 如果relation和entity是相同维度的就不需要设置double
    parser.add_argument('--emb_dim', default=5, type=int)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=False)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=False)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='学习率')
    parser.add_argument('--max_epoch', default=5, type=int, help='训练轮数')
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')

    parser.add_argument('--negative_sample_size', default=10, type=int, help='负采样个数')
    parser.add_argument('--negative_adversarial_sampling', action='store_true', default=True, help='是否使用自对抗负采样')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float, help='自对抗采样温度')
    parser.add_argument('--margin', default=12.0, type=float, help='The fixed margin in loss function. ')
    parser.add_argument('--regularization', default=0.0, type=float, help="regularization parameter")

    parser.add_argument('--uni_weight', action='store_true', default=True, help='False的话在算loss时使用重采样权重(word2vec)')

    parser.add_argument('-cpu', '--cpu_num', default=1, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str, help='加载config.json的路径')
    parser.add_argument('-save', '--save_path', default='models/demo_debug_model', type=str, help='模型保存路径')

    parser.add_argument('--warm_up_steps', default=None, type=int, help='预热步长，和学习率相关')

    parser.add_argument('--save_checkpoint_steps', default=5, type=int, help='每隔多少个step保存一次')
    parser.add_argument('--valid_steps', default=1, type=int)
    parser.add_argument('--log_steps', default=1, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=5, type=int, help='valid/test log every xx steps')

    parser.add_argument('--norm_flag', default=False, action='store_true', help='TransR和TransH用')

    parser.add_argument('--nentity', type=int, default=0, help='自动生成')
    parser.add_argument('--nrelation', type=int, default=0, help='自动生成')
    parser.add_argument('--train_triples', type=int, default=0, help='训练集三元组数，自动生成')

    return parser.parse_args()


if __name__ == '__main__':
    args = setup_parser()  # 设置参数
    if args.init_checkpoint:  # 不使用config设置的话就要定义数据集路径
        override_config(args)
    if args.save_path and not os.path.exists(args.save_path):  # 检查并创建保存路径
        os.makedirs(args.save_path)
    set_logger(args)  # 创建并写入日志文件

    entity2id, relation2id = get_id(args)

    logging.info('Dataset: %s' % args.dataset)
    logging.info('#实体数: %d' % args.nentity)
    logging.info('#关系数: %d' % args.nrelation)

    data_path = '../data/' + args.dataset + '/'
    train_triples = read_triple(args.dataset, os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(args.dataset, os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(args.dataset, os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    args.train_triples = len(train_triples)
    logging.info('#valid: %d' % len(valid_triples))
    logging.info('#test: %d' % len(test_triples))

    model = {
        'TransE': lambda: TransE.TransE(args),
        'TransR': lambda: TransR.TransR(args),
        'TransH': lambda: TransH.TransH(args),
        'RESCAL': lambda: RESCAL.RESCAL(args),
        'DistMult': lambda: DistMult.DistMult(args),
        'ComplEx': lambda: ComplEx.ComplEx(args),
        'RotatE': lambda: RotatE.RotatE(args),
        'SimplE': lambda: SimplE.SimplE(args),
    }[args.model_name]()

    logging.info('Model: %s' % args.model_name)

    # All true triples, 测试的时候使用
    all_true_triples = train_triples + valid_triples + test_triples

    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        model.to(args.device)

    if args.do_train:
        # Set training dataloader iterator
        # 设置训练集头实体的dataloader
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        # 尾实体
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(  # filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps  # warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用自定义学习率
        else:
            warm_up_steps = args.max_epoch // 2  # 如果没设置warm up参数，就将warm up参数设置到最大步数的一半

    if args.init_checkpoint:
        # Restore model from checkpoint directory 从检查点目录恢复模型
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_epoch = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        init_epoch = 0

    epoch = init_epoch

    logging.info('训练相关参数：')
    logging.info('init_epoch = %d' % init_epoch)
    logging.info('batch_size = %d' % args.batch_size)
    # logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('emb_dim = %d' % args.emb_dim)
    # logging.info('margin = %f' % args.margin)
    # logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    # if args.negative_adversarial_sampling:
    #     logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        logging.info('***************开始训练***************')
        logging.info('learning_rate = %f' % current_learning_rate)
        training_logs = []

        # Training Loop
        for epoch in range(init_epoch, args.max_epoch):
            log = model.train_step(model, optimizer, train_iterator, args)
            training_logs.append(log)

            if epoch >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at epoch %d' % (current_learning_rate, epoch))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=current_learning_rate)
                warm_up_steps = warm_up_steps * 3

            if epoch % args.save_checkpoint_steps == 0:  # 每save_checkpoint_steps步保存一次模型
                save_variable_list = {'epoch': epoch, 'current_learning_rate': current_learning_rate,
                                      'warm_up_steps': warm_up_steps}
                save_model(model, optimizer, save_variable_list, args)

            if epoch % args.log_steps == 0:  # 每log_steps步保存一次log, 把train log信息存在这里，然后training_log置空，进下一轮
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', epoch, metrics)
                training_logs = []

            if args.do_valid and epoch % args.valid_steps == 0:  # 每valid_steps步验证一下
                # logging.info('Evaluating on Valid Dataset...')
                # metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                # log_metrics('Valid', step, metrics)
                metrics = model.test_step(model, test_triples, all_true_triples, args)
                log_metrics('Test', epoch, metrics)

        save_variable_list = {'epoch': epoch, 'current_learning_rate': current_learning_rate,
                              'warm_up_steps': warm_up_steps}
        save_model(model, optimizer, save_variable_list, args)

        # if args.do_valid:
        #     logging.info('Evaluating on Valid Dataset...')
        #     metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        #     log_metrics('Valid', step, metrics)

        if args.do_test:
            logging.info('Evaluating on Test Dataset...')
            metrics = model.test_step(model, test_triples, all_true_triples, args)
            log_metrics('Test', epoch, metrics)

        # if args.evaluate_train:
        #     logging.info('Evaluating on Training Dataset...')
        #     metrics = model.test_step(model, train_triples, all_true_triples, args)
        #     log_metrics('Test', epoch, metrics)
