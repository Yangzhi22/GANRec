import numpy as np
from torch.utils.data import Dataset
import random
import networkx as nx


# def get_neigbors(g, node, depth=1):  # 获取 node 节点的邻接节点
#     output = {}
#     layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
#     nodes = [node]
#     for i in range(1, depth + 1):
#         output[i] = []
#         for x in nodes:
#             output[i].extend(layers.get(x, []))
#         nodes = output[i]
#     return output


class TrainGenerator(Dataset):
    def __init__(self, args_config, graph):
        self.args_config = args_config                  # 参数配置
        self.graph = graph                              # 协作知识图谱
        self.user_dict = graph.train_ui_user_dict       # 训练用户字典, 源代码
        self.exist_users = list(graph.exist_users)      # 存在的用户列表
        self.low_item_index = graph.item_range[0]       # 最小的项目索引
        self.high_item_index = graph.item_range[1]      # 最大的项目索引

    def __len__(self):
        return self.graph.train_num                     # 返回训练长度

    def __getitem__(self, index):
        out_dict = {}

        user_dict = self.user_dict                     # 用户字典
        # 随机选择一个用户
        u_id = random.sample(self.exist_users, 1)[0]   # 从存在的用户中随机采样一个用户
        out_dict["u_id"] = u_id

        # 随机选择一个正例项目
        pos_items = user_dict[u_id]                    # 所有的正例项目
        n_pos_items = len(user_dict[u_id])             # 正例项目长度

        pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_i_id = pos_items[pos_idx]

        out_dict["pos_i_id"] = pos_i_id

        neg_i_id = self.get_random_neg(pos_items, [])    # 源代码

        out_dict["neg_i_id"] = neg_i_id

        return out_dict

    def get_random_neg(self, pos_items, selected_items):           # 随机采样一个负例，源代码
        while True:
            neg_i_id = np.random.randint(
                low=self.low_item_index, high=self.high_item_index, size=1
            )[0]
            if neg_i_id not in pos_items and neg_i_id not in selected_items:
                break
        return neg_i_id


class TestGenerator(Dataset):                        # 测试集生成器，没有调用
    def __init__(self, args_config, graph):
        self.args_config = args_config
        self.users_to_test = list(graph.test_ui_user_dict.keys())

    def __len__(self):
        return len(self.users_to_test)

    def __getitem__(self, index):
        batch_data = {}

        u_id = self.users_to_test[index]
        batch_data["u_id"] = u_id

        return batch_data
