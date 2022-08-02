import numpy as np
import json
import os
import torch
import logging


def override_config(args):
    """
    Override model and data configuration，覆盖模型和数据设置
    """

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.emb_dim = argparse_dict['emb_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate  保存模型参数
    """

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)  # 把python对象转换成json对象生成一个fp的文件流

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(args.save_path, 'checkpoint')
    )
    if args.model_name == 'SimplE':
        ent_h_embs = model.ent_h_embs.cpu()
        np.save(os.path.join(args.save_path, 'ent_h_embs'), ent_h_embs)
        ent_t_embs = model.ent_t_embs.cpu()
        np.save(os.path.join(args.save_path, 'ent_t_embs'), ent_t_embs)
        rel_embs = model.rel_embs.cpu()
        np.save(os.path.join(args.save_path, 'rel_embs'), rel_embs)
        rel_inv_embs = model.rel_inv_embs.cpu()
        np.save(os.path.join(args.save_path, 'rel_inv_embs'), rel_inv_embs)
    elif args.model_name in ['TransR', 'TransH']:  # nn.Embedding
        entity_embedding = model.entity_embedding.weight.data.detach().cpu().numpy()
        np.save(os.path.join(args.save_path, 'entity_embedding'), entity_embedding)

        relation_embedding = model.relation_embedding.weight.data.detach().cpu().numpy()
        np.save(os.path.join(args.save_path, 'relation_embedding'), relation_embedding)

    else:  # nn.Parameter
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(os.path.join(args.save_path, 'entity_embedding'), entity_embedding)

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(os.path.join(args.save_path, 'relation_embedding'), relation_embedding)


def set_logger(args):
    """
    Write logs to checkpoint and console，将日志写入检查点和控制台
    """

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()  # 创建一个handler，用于输出到控制台
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs   log要记录的指标信息
    """
    if mode == 'Test':
        mrr = metrics['MRR']
        hit = [metrics['HITS@1'], metrics['HITS@3'], metrics['HITS@10']]
        metric_dict = {'MRR': mrr, 'hits@[1,3,10]': hit}
        logging.info('%s at epoch %d: %s' % (mode, step, metric_dict))
    else:
        for metric in metrics:
            logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))
