import torch
import torch.nn as nn


class Discriminator(nn.Module):                             # 决定器类，继承于 nn.Module
    def __init__(self, data_config, args_config):
        super(Discriminator, self).__init__()
        self.data_config = data_config                      # 数据设置
        self.args_config = args_config                      # 参数设置
        self.user_num = data_config['n_users']              # 用户数
        self.item_num = data_config['n_items']              # 项目数

        self.embed_size = args_config.emb_size              # 嵌入的维度
        self.regs = eval(args_config.regs)                  # 正则化参数

        self.all_embed = self._init_weight()                # 初始化所有的用户和项目的权重

    def _init_weight(self):                                 # 初始化权重参数函数

        # [用户数 + 项目数, 嵌入维度]
        all_embed = nn.Parameter(torch.FloatTensor(self.user_num + self.item_num, self.embed_size))

        if self.args_config.pretrain_d:                     # 如果使用了预训练的识别器
            all_embed.data = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)
        return all_embed

    def forward(self, user, pos_item, neg_item):           # 前向传播
        u_e = self.all_embed[user]                         # 用户嵌入
        pos_e = self.all_embed[pos_item]                   # 正例项目嵌入
        neg_e = self.all_embed[neg_item]                   # 负例项目嵌入

        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)      # l2 损失
        reg_loss = self.regs * reg_loss

        pos_scores = torch.sum(u_e * pos_e, dim=1)         # 正例得分
        neg_scores = torch.sum(u_e * neg_e, dim=1)         # 负例得分

        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))    # 源码
        # bpr_loss = torch.log(torch.sigmoid(neg_scores - pos_scores))
        bpr_loss = -torch.mean(bpr_loss)
        return bpr_loss, reg_loss                          # 返回损失

    @staticmethod
    def _l2_loss(t):                                       # 定义静态的 l2 损失函数
        return torch.sum(t ** 2) / 2

    def get_reward(self, user, pos_item, neg_item):        # 定义奖励函数
        u_e = self.all_embed[user]                         # 用户嵌入
        pos_e = self.all_embed[pos_item]                   # 项目的正例嵌入
        neg_e = self.all_embed[neg_item]                   # 项目的负例嵌入

        # neg_scores = torch.sum(u_e * pos_e, dim=-1)        # 尚未搞清楚有啥好处
        # ij = torch.sum(pos_e * neg_e, dim=-1)
        # reward = neg_scores + ij
        ###################################################################
        pos_scores = torch.sum(u_e * pos_e, dim=-1)  # 正例得分
        neg_scores = torch.sum(u_e * neg_e, dim=-1)  # 负例得分
        reward = pos_scores + neg_scores             # 使用正例得分减去负例得分作为奖励
        ###################################################################
        return reward

    # def item_rank(self, users, neg_items):          # 定义一个负例项目打分函数
    #     u_e = self.all_embed[users]                 # 用户嵌入
    #     neg_i_e = self.all_embed[neg_items]         # 负例项目的嵌入   torch.Size([1024, 32, 64])
    #     u_e = u_e.unsqueeze(dim=1)                  # 在用户嵌入维度为 1 的地方扩张一个维度为 1 的维度 torch.Size([1024, 1, 64])
    #     ranking = torch.sum(u_e * neg_i_e, dim=2)   # 在维度等于 2 的地方相乘
    #     ranking = ranking.squeeze()                 # 压缩维度为 1 的维度
    #     return ranking

    ####################################################################################################################
    def item_rank(self, users, pos_items, neg_items):  # 定义一个负例项目打分函数
        u_e = self.all_embed[users]  # 用户嵌入
        pos_i_e = self.all_embed[pos_items]
        neg_i_e = self.all_embed[neg_items]  # 负例项目的嵌入   torch.Size([1024, 32, 64])
        u_e = u_e.unsqueeze(dim=1)  # 在用户嵌入维度为 1 的地方扩张一个维度为 1 的维度 torch.Size([1024, 1, 64])
        ranking = torch.sum(u_e * (pos_i_e - neg_i_e), dim=2)    # 在维度等于 2 的地方相乘（源码）
        # ranking = torch.sum(u_e * (neg_i_e - pos_i_e), dim=2)    # 在维度等于 2 的地方相乘
        ranking = ranking.squeeze()                              # 压缩维度为 1 的维度
        return ranking

    ####################################################################################################################

    def __str__(self):
        return "Using BPR loss as the discriminator, the embedding length is: {}".format(self.args_config.emb_size)