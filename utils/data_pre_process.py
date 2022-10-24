"""
# 作者： 杨志
# 时间： 2020年11月19日

"""

import collections
import networkx as nx
from tqdm import tqdm
import numpy as np


class UIData(object):     # 用户-项目二部图的交互数据
    def __init__(self, args_config):
        self.args_config = args_config
        path = args_config.data_path + args_config.dataset                    # 数据集的路径设置
        # print("path:", path)
        train_file = path + "/train.dat"                                      # 训练集文件
        test_file = path + "/test.dat"                                        # 测试集文件
        # print("train_file:", train_file)
        self.train_ui_data = self._generate_ui_interactions(train_file)       # 从训练集中生成用户-项目的交互
        self.test_ui_data = self._generate_ui_interactions(test_file)         # 从训练集中生成用户-项目的交互

        self.train_ui_user_dict, self.test_ui_user_dict = self._generate_user_dict()   # 生成训练集和测试集的用户项目交互字典

        self.exist_users = list(self.train_ui_user_dict.keys())               # 存在的所有用户
        self._statistic_interactions_info()                                   # 打印用户-项目交互二部图的交互信息

    @staticmethod       # 从训练集和测试集中读取用户-项目的交互数据
    def _generate_ui_interactions(file_name):                                 # 生成用户-项目交互的函数
        ui_mat = list()                                                       # 用户-项目交互列表
        with open(file_name, "r") as files:
            lines = files.readlines()                                         # 读取所有数据
            for every_line in lines:
                temps = every_line.strip()
                inters = [int(i) for i in temps.split(" ")]                   # 列表

                user_id, pos_ids = inters[0], inters[1:]                      # 用户id, 产生交互的所有项目
                pos_ids = list(set(pos_ids))                                  # 去重
                for item_id in pos_ids:
                    ui_mat.append([user_id, item_id])
        return np.array(ui_mat)

    def _generate_user_dict(self):                                            # 生成用户-项目的交互字典
        def _generate_dict(ui_mat):                                           # 生成字典函数
            user_dict = dict()
            for user_id, item_id in ui_mat:
                if user_id not in user_dict.keys():
                    user_dict[user_id] = list()
                user_dict[user_id].append(item_id)

            return user_dict

        users_num = max(max(self.train_ui_data[:, 0]), max(self.test_ui_data[:, 0])) + 1  # 求出所有的用户数

        # 重构项目的范围，从 [0, #items) 到 [#num_users, #num_users + #items)
        self.train_ui_data[:, 1] = self.train_ui_data[:, 1] + users_num
        self.test_ui_data[:, 1] = self.test_ui_data[:, 1] + users_num

        train_user_dict = _generate_dict(self.train_ui_data)
        test_user_dict = _generate_dict(self.test_ui_data)

        return train_user_dict, test_user_dict

    def _statistic_interactions_info(self):                                      # 统计项目二部图的交互信息
        def _id_range(train_ui_mat, test_ui_mat, idx):                           # idx 为用户或项目索引
            min_id = min(min(train_ui_mat[:, idx]), min(test_ui_mat[:, idx]))    # 计算最小 id
            max_id = max(max(train_ui_mat[:, idx]), max(test_ui_mat[:, idx]))    # 计算最大 id

            num_id = max_id - min_id + 1                                         # 计算 id 范围
            return (min_id, max_id), num_id                                      # 返回 (最小id， 最大id), id数目

        self.user_range, self.users_num = _id_range(self.train_ui_data, self.test_ui_data, idx=0)
        self.item_range, self.items_num = _id_range(self.train_ui_data, self.test_ui_data, idx=1)
        self.train_num = len(self.train_ui_data)                                 # 训练长度
        self.test_num = len(self.test_ui_data)                                   # 测试长度

        print("-" * 50)
        print("-     user_range: (%d, %d)" % (self.user_range[0], self.user_range[1]))      # 打印用户范围
        print("-     item_range: (%d, %d)" % (self.item_range[0], self.item_range[1]))      # 打印项目范围
        print("-     train_number: %d" % self.train_num)                                      # 打印训练数
        print("-     test_number: %d" % self.test_num)                                       # 打印测试数
        print("-     users_number: %d" % self.users_num)                                        # 所有的用户数
        print("-     items_number: %d" % self.items_num)                                        # 所有的项目数
        print("-" * 50)


class KGData(object):           # 知识图谱的交互数据
    def __init__(self, args_config, entity_start_id=0, relation_start_id=0):
        self.args_config = args_config
        self.entity_start_id = entity_start_id                                   # 实体开始id
        self.relation_start_id = relation_start_id                               # 关系开始id
        path = args_config.data_path + args_config.dataset                       # 数据集的路径设置
        # print("path:", path)
        kg_file = path + "/kg_final.txt"                                         # 知识图谱文件

        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)  # 加载知识图谱数据
        self._statistic_kg_info()

    def _load_kg(self, filename):                                                # 加载知识图谱的函数
        def _remap_kg_id(old_kg):                                                # 重构实体的范围，需要加上用户数
            new_kg = old_kg.copy()
            new_kg[:, 0] = old_kg[:, 0] + self.entity_start_id
            new_kg[:, 2] = old_kg[:, 2] + self.entity_start_id
            new_kg[:, 1] = old_kg[:, 1] + self.relation_start_id                 # 考虑用户-项目二部图中的交互与被交互关系
            return new_kg

        def _construct_kg_dict(kg_data):                                         # 构建知识图谱字典
            kg_dict = collections.defaultdict(list)
            relation_dict = collections.defaultdict(list)

            for head, relation, tail in kg_data:
                kg_dict[head].append((tail, relation))                # {头实体：[(尾实体1， 关系1), (尾实体1， 关系1), ...]}
                relation_dict[relation].append((head, tail))          # {关系：[(头实体1， 尾实体1), (头实体1， 尾实体1), ...]}
            return kg_dict, relation_dict

        # 加载项目知识图谱 <item, has-aspect, entity>
        can_kg_np = np.loadtxt(filename, dtype=np.int32)
        can_kg_np = np.unique(can_kg_np, axis=0)                                 # 去重

        can_kg_np = _remap_kg_id(can_kg_np)                                      # 重构知识图谱的范围

        # 构建项目知识图谱的逆向交互关系
        inv_kg_np = can_kg_np.copy()
        inv_kg_np[:, 0] = can_kg_np[:, 2]
        inv_kg_np[:, 2] = can_kg_np[:, 0]
        inv_kg_np[:, 1] = can_kg_np[:, 1] + max(can_kg_np[:, 1]) + 1

        # 将正向和逆向的项目知识图谱拼接在一起
        all_kg_data = np.concatenate((can_kg_np, inv_kg_np), axis=0)
        all_kg_dict, all_relation_dict = _construct_kg_dict(all_kg_data)
        return all_kg_data, all_kg_dict, all_relation_dict

    def _statistic_kg_info(self):                                       # 统计知识图谱的交互信息
        def _id_range(kg_mat, idx):
            min_id = min(min(kg_mat[:, idx]), min(kg_mat[:, 2 - idx]))  # 最小id
            max_id = max(max(kg_mat[:, idx]), max(kg_mat[:, 2 - idx]))  # 最大id
            num_id = max_id - min_id + 1                                # id范围
            return (min_id, max_id), num_id                             # 返回 (最小id， 最大id), id数目

        self.entity_range, self.entity_num = _id_range(self.kg_data, idx=0)      # 求解知识图谱中的实体范围和实体数
        self.relation_range, self.relation_num = _id_range(self.kg_data, idx=1)  # 求解知识图谱中的关系范围和关系数
        self.kg_triples_num = len(self.kg_data)

        print("-" * 50)
        print("-     entity_range: (%d, %d)" % (self.entity_range[0], self.entity_range[1]))      # 打印实体范围
        print("-     relation_range: (%d, %d)" % (self.relation_range[0], self.relation_range[1]))  # 打印关系范围
        print("-     entities_number: %d" % self.entity_num)                                           # 打印实体数
        print("-     relations_number: %d" % self.relation_num)                                         # 打印关系数
        print("-     triples_number: %d" % self.kg_triples_num)                               # 所有的知识图三元组数
        print("-" * 50)


class CKGData(UIData, KGData):                                    # 协作知识图谱（CKG），继承于 UIData类和 KGData类
    def __init__(self, args_config):
        UIData.__init__(self, args_config=args_config)            # UIData类的初始化
        KGData.__init__(
            self,
            args_config=args_config,
            entity_start_id=self.users_num,
            relation_start_id=2)                                  # KGData类的初始化
        self.args_config = args_config
        self.ckg_graph, self.ckg_graph1 = self._combine_ui_kg()  # 构建用户-项目二部图和知识图谱的联合体（协作知识图谱）

    def _combine_ui_kg(self):
        ui_data = self.train_ui_data                              # 训练的用户-项目交互数据
        kg_data = self.kg_data                                    # 知识图谱数据

        # 连接用户-项目交互数据和知识图谱数据:
        # ... 用户实体的范围 [0, #users)
        # ... 项目实体的范围 [#users, #users + #items)
        # ... 其他实体的范围 [#users + #items, #users + #entities)
        # ... 关系的范围 [0, 2 + 2 * #kg relations), 包括交互和被交互的关系.
        ckg_graph = nx.MultiDiGraph()                             # 构建图数据
        ckg_graph1 = nx.MultiDiGraph()                            # 构建图数据
        print("Loading user-item interaction triples ...")
        for user_id, item_id in tqdm(ui_data, ascii=True):
            ckg_graph.add_edges_from([(user_id, item_id)], relation_id=0)
            ckg_graph.add_edges_from([(item_id, user_id)], relation_id=1)
            ckg_graph1.add_edges_from([(user_id, item_id)], relation_id=0)
            ckg_graph1.add_edges_from([(item_id, user_id)], relation_id=1)

        print("\nLoading KG interaction triples ...")
        for head_id, relation_id, tail_id in tqdm(kg_data, ascii=True):
            ckg_graph.add_edges_from([(head_id, tail_id)], relation_id=relation_id)

        return ckg_graph, ckg_graph1

