import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class GraphConv(nn.Module):                                    # 图神经网络类
    """
    图神经网络
    输入：知识图谱的嵌入矩阵（原来的为实体及其邻接矩阵的嵌入矩阵）
    输出：知识图谱中实体的图卷积嵌入
    """
    def __init__(self, in_channel, out_channel, config):
        super(GraphConv, self).__init__()
        self.config = config                          # 参数设置
        # self.conv1 = geometric.nn.GCNConv(in_channel[0], out_channel[0])  # 第一层卷积  （GCN图卷积神经网络）
        # self.conv2 = geometric.nn.GCNConv(in_channel[1], out_channel[1])  # 第二层卷积
        self.conv1 = SAGEConv(in_channel[0], out_channel[0])    # 第一层卷积  （GraphSage图卷积神经网络）
        self.conv2 = SAGEConv(in_channel[1], out_channel[1])    # 第二层卷积
        # self.conv1 = geometric.nn.GATConv(in_channel[0], out_channel[0])  # 第一层卷积  （GAT图注意力神经网络）
        # self.conv2 = geometric.nn.GATConv(in_channel[1], out_channel[1])  # 第二层卷积

    def forward(self, x, edge_indices):        # 前向传播函数，输入为节点的嵌入矩阵和节点的邻接节点的嵌入矩阵
        x = self.conv1(x, edge_indices)        # 第一次卷积操作
        x = F.leaky_relu(x)                    # 线性激活函数，有研究显示没有多大作用，可以试试
        x = F.dropout(x)                       # Dropout操作，防止过拟合

        x = self.conv2(x, edge_indices)        # 第二次卷积操作
        x = F.dropout(x)
        x = F.normalize(x)                     # 标准化操作

        return x


class Generator(nn.Module):                                    # 生成器类
    """
    通过图神经嵌入和知识图谱生成符合识别器的负例
    输入：用户， 正例项目， 知识图谱嵌入
    输出：能够欺骗识别器的负例
    """
    def __init__(self, discriminator, params, config):
        super(Generator, self).__init__()
        self.discriminator = discriminator                      # 识别器参数
        self.params = params                                    # 数据配置
        self.config = config                                    # 参数配置

        in_channel = eval(config.in_channel)                    # 图神经网络的输入形状 [64, 32]
        out_channel = eval(config.out_channel)                  # 图神经网络的输出形状 [32, 64]

        self.gcn = GraphConv(in_channel, out_channel, config)   # 调用图神经网络学习用户和项目的表示

        self.n_entities = params['n_nodes']                     # 实体数，即所有节点数
        self.item_range = params['item_range']                  # 项目范围
        self.input_channel = in_channel                         # 图神经网络的输入形状
        self.entity_embedding = self._initialize_weight(self.n_entities, self.input_channel)        # 实体嵌入权重初始化函数

    def _initialize_weight(self, n_entities, input_channel):       # 实体嵌入函数（通过 xavier 初始化）
        """ 实体包含知识图谱中的项目和其他实体 """
        if self.config.pretrain_g:
            kg_embedding = self.params['kg_embedding']
            entity_embedding = nn.Parameter(kg_embedding)
        else:
            entity_embedding = nn.Parameter(torch.FloatTensor(n_entities, input_channel[0]))
            nn.init.xavier_uniform_(entity_embedding)

            if self.config.freeze_g:
                entity_embedding.requires_grad = False

        return entity_embedding

    def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch['u_id']                               # 用户
        pos = data_batch['pos_i_id']                             # 正例项目

        self.edges = self.build_edge(edge_matrix)                # 构建边
        x = self.entity_embedding  # 实体嵌入
        gcn_embedding = self.gcn(x, self.edges.t().contiguous())  # 实体和边的图卷积嵌入

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)  # 负例列表
        prob_list = torch.tensor([], device=adj_matrix.device)                   # 概率列表

        k = self.config.k_step                                                   # k = 2 探索步骤
        assert k > 0                                                             # 断言 k > 0

        for _ in range(k):
            """ 基于知识图谱采样负例项目 """
            one_hop, one_hop_logits = self.kg_step(gcn_embedding, pos, users, adj_matrix, 1, step=1)    # 一跳的候选负例及其概率
            candidate_neg, two_hop_logits = self.kg_step(gcn_embedding, one_hop, users, adj_matrix, 1, step=2)   # 两跳的候选负例及其概率
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)                 # 过滤掉不在项目范围内的实体

            # # 使用剪枝策略过滤掉不太可能的负例
            #########################################################################################################
            one_hop_pos, one_hop_logits_pos = self.kg_step(gcn_embedding, pos, users, adj_matrix, 2, step=2)  # 一跳的候选负例及其概率
            one_hop_pos = self.filter_entity(one_hop_pos, self.item_range)  # 过滤掉不在项目范围内的实体
            good_neg, good_logits = self.prune_step(self.discriminator, users, one_hop_pos,
                                                    candidate_neg, two_hop_logits)  # 剪枝策略
            #########################################################################################################
            good_logits = good_logits + one_hop_logits                                         # 源代码

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])                            # 负例列表
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])                       # 概率列表

            pos = good_neg

        return neg_list, prob_list

    def build_edge(self, edge_matrix):                                            # 基于邻接矩阵构建边
        sample_edge = self.config.edge_threshold                                  # edge_threshold = 64
        edge_matrix = edge_matrix
        n_node = edge_matrix.size(0)                                              # 节点数
        node_index = (torch.arange(n_node, device=edge_matrix.device, dtype=torch.long).unsqueeze(1).repeat(1, sample_edge).flatten())
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges                                                              # 返回边

    def kg_step(self, gcn_embedding, pos, users, adj_matrix, explore_flag, step):                # GNN探索知识图谱的步数

        """用知识图谱的图卷积嵌入决定候选的负例的项目"""
        u_e = gcn_embedding[users]                                               # 用户的图卷积嵌入 [batch_size, 64]
        u_e = u_e.unsqueeze(dim=2)                                               # [batch_size, 64, 1]

        pos_e = gcn_embedding[pos]                                              # 正例项目的图卷积嵌入 [batch_size, 64]
        pos_e = pos_e.unsqueeze(dim=1)                                          # [batch_size, 1, 64]

        ################################################################################################################
        if explore_flag == 1:
            one_hop = adj_matrix[pos]           # 邻接矩阵的一跳邻居 [batch_size, 128]
            i_e = gcn_embedding[one_hop]        # 一跳邻居的图卷积嵌入 [batch_size, 128, 64]
        else:
            one_hop = adj_matrix[users]
            i_e = gcn_embedding[one_hop]
        ################################################################################################################
        p_entity = F.leaky_relu(pos_e * i_e)                                      # 实体的嵌入（源代码）
        p = torch.matmul(p_entity, u_e)                                           # 实体嵌入与用户的匹配得分
        p = p.squeeze()
        logits = F.softmax(p, dim=1)                                              # 最大的概率分布

        """ 基于最大概率分布采样负例项目 """
        batch_size = logits.size(0)                                               # 采样的长度
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample                                            # 采样 32 个负例
            _, indices = torch.sort(logits, descending=True)                      # 按行从大到小排序
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)      # 行索引 id

        candidate_neg = one_hop[row_id, nid].squeeze()                            # 候选负例
        candidate_prob = torch.log(logits[row_id, nid]).squeeze()                 # 候选负例对应的概率
        return candidate_neg, candidate_prob

    @staticmethod
    # def prune_step(discriminator, users, negs, logits):
    #     with torch.no_grad():
    #         ranking = discriminator.item_rank(users, negs)
    #################################################################################################################
    def prune_step(discriminator, users, pos, negs, logits):
        with torch.no_grad():
            ranking = discriminator.item_rank(users, pos, negs)
        indices = torch.argmin(ranking, dim=1)
    #################################################################################################################

        """ 基于用户和负例项目的相似性获得高质量的负例项目 """
        # indices = torch.argmax(ranking, dim=1)                                 # 按行求最大的索引
        batch_size = negs.size(0)                                                # 负例个数
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)       # 行 id
        indices = indices.unsqueeze(1)                                           # 列索引

        good_neg = negs[row_id, indices].squeeze()
        good_logits = logits[row_id, indices].squeeze()

        return good_neg, good_logits

    @staticmethod
    def filter_entity(candidate_neg, item_range):                                # 过滤掉不属于项目范围内的实体
        random_neg = torch.randint(                                              # 随机产生一些负例
            int(item_range[0]), int(item_range[1] + 1), candidate_neg.size(), device=candidate_neg.device
        )
        candidate_neg[candidate_neg > item_range[1]] = random_neg[candidate_neg > item_range[1]]
        candidate_neg[candidate_neg < item_range[0]] = random_neg[candidate_neg < item_range[0]]

        return candidate_neg