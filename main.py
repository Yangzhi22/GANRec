# -*- coding: utf8 -*-

"""
# 作者： 杨志
# 时间： 2020年11月19日

"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import torch
from config.parser import parse_args
from utils.data_pre_process import CKGData
from copy import deepcopy
from utils.build import build_dataset_loader
from modules.discriminator.Discriminator import Discriminator
from modules.generator.Generator import Generator
from utils.utils import early_stopping, print_dict
from test1 import test_v2
from time import time
from tqdm import tqdm
from prettytable import PrettyTable
import pickle as pkl
import networkx as nx


def train_one_epoch(
    discriminator,              # 识别器参数
    generator,                  # 生成器参数
    train_loader,               # 训练集加载器
    discriminator_optimizer,    # 识别器优化
    generator_optimizer,        # 生成器优化
    adj_matrix,                 # 邻接矩阵
    edge_matrix,                # 边矩阵
    train_data,                 # 训练数据
    cur_epoch,                  # 当前步骤
    avg_reward,                 # 平均奖励
):

    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    tbar = tqdm(train_loader, ascii=True)
    num_batch = len(train_loader)
    # print("\nnum_batch = ", num_batch)
    for batch_data in tbar:

        tbar.set_description("Epoch {}".format(cur_epoch))

        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train discriminator using negtive item provided by generator"""
        pos = batch_data["pos_i_id"]  # 正例
        users = batch_data["u_id"]  # 用户 id
        loss_batch, base_loss_batch, reg_loss_batch = 0., 0., 0.
        for i in range(12):  # 生成对抗网络的思想进行训练
            discriminator_optimizer.zero_grad()  # 一个epoch的推荐器导数清零

            neg = batch_data["neg_i_id"]  # 负例

            selected_neg_items_list, _ = generator(batch_data, adj_matrix, edge_matrix)  # 生成器生成的负例列表
            selected_neg_items = selected_neg_items_list[-1, :]  # 取最后生成器生成的负例
            train_set = train_data[users]    # 批数据的用户训练数据  tensor = [[76658, 76245, ...], ... , [76658, 76245, ...]]
            in_train = torch.sum(
                selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
            ).bool()
            selected_neg_items[in_train] = neg[in_train]                         # [1, batch_size]
            base_loss_batch, reg_loss_batch = discriminator(users, pos, selected_neg_items)
            loss_batch = base_loss_batch + reg_loss_batch

            loss_batch.backward()
            discriminator_optimizer.step()

        # """Train discriminator using negtive item provided by generator"""
        # discriminator_optimizer.zero_grad()                                             # 一个epoch的推荐器导数清零
        # 
        # neg = batch_data["neg_i_id"]                                                    # 负例
        # pos = batch_data["pos_i_id"]                                                    # 正例
        # users = batch_data["u_id"]                                                      # 用户 id
        # selected_neg_items_list, _ = generator(batch_data, adj_matrix, edge_matrix)     # 生成器生成的负例列表
        # selected_neg_items = selected_neg_items_list[-1, :]                             # 取最后生成器生成的负例
        # train_set = train_data[users]             # 批数据的用户训练数据  tensor = [[76658, 76245, ...], ... , [76658, 76245, ...]]
        # in_train = torch.sum(
        #     selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
        # ).bool()
        # selected_neg_items[in_train] = neg[in_train]
        # base_loss_batch, reg_loss_batch = discriminator(users, pos, selected_neg_items)
        # loss_batch = base_loss_batch + reg_loss_batch
        # loss_batch.backward()
        # discriminator_optimizer.step()

        """Train generator network"""
        generator.zero_grad()
        selected_neg_items_list, selected_neg_prob_list = generator(
            batch_data, adj_matrix, edge_matrix
        )

        with torch.no_grad():
            reward_batch = discriminator.get_reward(users, pos, selected_neg_items_list)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        # batch_size = reward_batch.size(1)
        # n = reward_batch.size(0) - 1
        # R = torch.zeros(batch_size, device=reward_batch.device)
        # reward = torch.zeros(reward_batch.size(), device=reward_batch.device)
        #
        # gamma = args_config.gamma
        #
        # for i, r in enumerate(reward_batch.flip(0)):
        #     R = r + gamma * R
        #     reward[n - i] = R

        reinforce_loss = -1 * torch.sum(reward_batch * selected_neg_prob_list)
        reinforce_loss.backward()
        generator_optimizer.step()

        """record loss in an epoch"""
        loss += loss_batch
        reg_loss += reg_loss_batch
        base_loss += base_loss_batch

    avg_reward = epoch_reward / num_batch
    train_res = PrettyTable()
    train_res.field_names = ["Epoch", "Loss", "BPR-Loss", "Regulation", "AVG-Reward"]
    train_res.add_row(
        [cur_epoch, loss.item(), base_loss.item(), reg_loss.item(), avg_reward.item()]
    )
    print(train_res)

    return loss, base_loss, reg_loss, avg_reward


def build_sampler_graph(n_nodes, edge_threshold, adj_graph_dict, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)        # 邻接矩阵 [节点数, 128]
    edge_matrix = torch.zeros(n_nodes, edge_threshold)           # 边矩阵 [节点数, 64]

    """sample neighbors for each node"""
    for node in tqdm(graph.ckg_graph1.nodes, ascii=True, desc="Build sampling matrix"):  # ckg_graph1 为二部图, ckg_graph为知识图谱
        if node in adj_graph_dict.keys():
            # if node in graph.train_ui_user_dict.keys():              # 三阶判断
            #     neighbors = adj_graph_dict[node]
            # else:
            #     neighbors = list(graph.ckg_graph.neighbors(node))
            neighbors = adj_graph_dict[node] + list(graph.ckg_graph1.neighbors(node))
        else:
            neighbors = list(graph.ckg_graph1.neighbors(node))
    # for node in tqdm(graph.ckg_graph1.nodes, ascii=True, desc="Build sampler matrix"):    # 一阶判断，用户二部图
    #     neighbors = list(graph.ckg_graph1.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(graph.item_range[0], graph.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        """concatenate sampled edge with random edge"""
        sampled_edge += random.sample(
            range(graph.item_range[0], graph.item_range[1] + 1), edge_threshold
        )

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.long().cuda()
        edge_matrix = edge_matrix.long().cuda()

    return adj_matrix, edge_matrix


def build_train_data(train_mat):          # 构建训练矩阵，形状为 [用户数, 最大的项目数]，其中不足最大长度的补-1
    user_num = max(train_mat.keys()) + 1  # 最大的训练用户数
    print("Maximum number of training users: ", user_num)
    true_num = max([len(i) for i in train_mat.values()])  # 最大的训练项目数
    print("Maximum number of training items: ", true_num)
    train_data = torch.zeros(user_num, true_num)  # 构建训练矩阵

    for i in train_mat.keys():
        true_list = train_mat[i]
        true_list += [-1] * (true_num - len(true_list))  # 低于最大项目数的位置补 -1
        train_data[i] = torch.tensor(true_list, dtype=torch.long)

    return train_data


def train(train_loader, test_loader, graph, data_config, args_config, adj_graph_dict):
    train_mat = graph.train_ui_user_dict             # 构建训练数据矩阵，多出的部分补-1
    train_data = build_train_data(train_mat)         # 调用函数构建训练矩阵
    if args_config.pretrain_d:
        print("\nLoading model: {}".format(args_config.data_path + args_config.model_path))
        paraments = torch.load(args_config.data_path + args_config.model_path)            # 加载预训练的模型
        all_embed = torch.cat((paraments['user_para'], paraments['item_para']))           # torch.cat 函数的拼接维度默认为0
        data_config['all_embed'] = all_embed

    discriminator = Discriminator(data_config=data_config, args_config=args_config)       # 配置识别器
    generator = Generator(discriminator, data_config, args_config)                        # 依据识别器，配置生成器

    if torch.cuda.is_available():
        train_data = train_data.cuda().long()
        discriminator = discriminator.cuda()
        generator = generator.cuda()

        print("\nGenerator: {}".format(str(generator)))
        print("Discriminator: {}\n".format(str(discriminator)))

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args_config.dlr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args_config.glr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step, cur_best_pre_0, avg_reward = 0, 0.0, 0
    t0 = time()

    for epoch in range(args_config.epoch):
        # if epoch % args_config.adj_epoch == 0:
        #     adj_matrix, edge_matrix = build_sampler_graph(
        #         data_config['n_nodes'], args_config.edge_threshold, graph
        #     )
        ################################################################################################################
        if epoch % args_config.adj_epoch == 0:
            adj_matrix, edge_matrix = build_sampler_graph(
                data_config['n_nodes'], args_config.edge_threshold, adj_graph_dict, graph
            )
        ################################################################################################################
        cur_epoch = epoch + 1
        loss, base_loss, reg_loss, avg_reward = train_one_epoch(
            discriminator,
            generator,
            train_loader,
            discriminator_optimizer,
            generator_optimizer,
            adj_matrix,                       # [n_nodes, 128]
            edge_matrix,                      # [n_nodes, 64]
            train_data,                       # [n_users, max_interaction]
            cur_epoch,
            avg_reward,
        )

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                ret = test_v2(discriminator, args_config.Ks, graph)

            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])

            print_dict(ret)
            # path = './Data/result-' + args_config.dataset + '-adj_three_order1.txt'
            # with open(path, "a+", encoding="UTF-8") as f:
            #     f.writelines("\n epoch = %s \n %s" % (cur_epoch, ret))
            #     print("Write down!")

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=args_config.flag_step,
            )

            if should_stop:
                break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = (
            "Best Iter=[%d]@[%.1f]\n recall=[%s] \n precision=[%s] \n hit=[%s] \n ndcg=[%s]"
            % (
                idx,
                time() - t0,
                "\t".join(["%.5f" % r for r in recs[idx]]),
                "\t".join(["%.5f" % r for r in pres[idx]]),
                "\t".join(["%.5f" % r for r in hit[idx]]),
                "\t".join(["%.5f" % r for r in ndcgs[idx]]),
            )
    )
    print(final_perf)
    # path = './Data/final_result-' + args_config.dataset + '-adj_three_order1.txt'
    # with open(path, "w", encoding="UTF-8") as f:
    #     f.writelines(final_perf)
    #     print("Write down!")


def get_neigbors(g, node, depth=1):                           # 获取 node 节点的邻接节点, 可用于求高阶可达节点
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output


if __name__ == "__main__":
    """ 固定随机种子"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    """ 初始化参数和数据集 """
    args_config = parse_args()                      # 模块测试代码
    file_name = './Data/CKG-' + args_config.dataset + '.pkl'
    try:
        with open(file_name, 'rb') as files:
            print('Loading CKG ...')
            CKG = pkl.load(files)
    except EOFError or FileNotFoundError:
        print('The CKG has not been saved yet, it may take some time to build！')
        CKG = CKGData(args_config)
        files = open(file_name, 'wb')
        pkl.dump(CKG, files)
    finally:
        with open(file_name, 'rb') as files:
            print('Opening CKG ...')
            CKG = pkl.load(files)

    CKG._statistic_interactions_info()
    CKG._statistic_kg_info()

    ####################################################################################################################
    file_name1 = './Data/adj_graph_dict-' + args_config.dataset + '.pkl'
    try:
        with open(file_name1, 'rb') as files:
            print('Loading adj_graph_dict ...')
            adj_graph_dict = pkl.load(files)
    except EOFError or FileNotFoundError:
        print('The adj_graph_dict has not been saved yet, it may take some time to build!')
        adj_graph_dict = dict()
        for node in tqdm(CKG.ckg_graph1.nodes, ascii=True, desc="High order interaction"):  # 二部图中的每一个节点
            adj_graph_dict[node] = get_neigbors(CKG.ckg_graph1, node, depth=3)[3]  # 每个节点的三阶邻居节点
        files = open(file_name1, 'wb')
        pkl.dump(adj_graph_dict, files)
    finally:
        with open(file_name1, 'rb') as files:
            print('Opening adj_graph_dict ...')
            adj_graph_dict = pkl.load(files)
    ####################################################################################################################

    # data_config = {  # 数据配置，协作知识图谱
    #     "n_users": CKG.users_num,  # 用户数
    #     "n_items": CKG.items_num,  # 项目数
    #     "n_relations": CKG.relation_num + 2,  # 知识图谱中的关系数加2 （0和1，来源于用户-项目二部图）
    #     "n_entities": CKG.entity_num,         # 实体数 = 头实体数 + 尾实体数
    #     "n_nodes": CKG.entity_range[1] + 1,   # 所有节点的数目
    #     "item_range": CKG.item_range,         # 项目的范围，一个二元组
    # }

    data_config = {  # 数据配置，只有用户-项目二部图
        "n_users": CKG.users_num,  # 用户数
        "n_items": CKG.items_num,  # 项目数
        "n_relations": 2,  # 知识图谱中的关系数加2 （0和1，来源于用户-项目二部图）
        "n_entities": CKG.users_num + CKG.items_num,  # 实体数 = 头实体数 + 尾实体数
        "n_nodes": CKG.item_range[1] + 1,  # 所有节点的数目
        "item_range": CKG.item_range,  # 项目的范围，一个二元组
    }

    print("\nOpening CKG ...")
    try:
        with open(file_name, 'rb') as files:
            graph = pkl.load(files)
    except EOFError or FileNotFoundError:
        print("File reading exception!")

    print("User num:", CKG.users_num)
    print("Train batch_size: ", args_config.batch_size)

    train_loader, test_loader = build_dataset_loader(args_config=args_config, graph=graph)    # 训练集和测试集加载器

    train(
        train_loader=train_loader,
        test_loader=test_loader,
        graph=CKG,
        data_config=data_config,
        args_config=args_config,
        adj_graph_dict=adj_graph_dict,
    )

