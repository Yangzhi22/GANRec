import torch

import numpy as np
from tqdm import tqdm

path = "./Dataset/result.txt"


def get_score(model, n_users, n_items, train_user_dict, s, t):
    u_e, i_e = torch.split(model.all_embed, [n_users, n_items], dim=0)

    u_e = u_e[s:t, :]
    score_matrix = torch.matmul(u_e, i_e.t())
    for u in range(s, t):
        if n_users == 23566:
            pos = list(set(train_user_dict[u]))
        else:
            pos = train_user_dict[u]

        idx = pos.index(-1) if -1 in pos else len(pos)
        score_matrix[u-s][pos[:idx] - n_users] = -1e5

    return score_matrix


def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg

########################################################################################################################
#
# def dcg_at_k(r, k):                                           # 与 KGAT、 IRGAN 采用同样的测评方式
#     r = np.asfarray(r)[:k]
#     return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#
#
# def ndcg_at_k(r, k):
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k) / dcg_max
#
#
# def cal_ndcg(topk, test_set, num_pos, k):
#     r = []
#     for i in set(topk):
#         if i in test_set:
#             r.append(1)
#         else:
#             r.append(0)
#     ndcg = ndcg_at_k(r, k)
#     return ndcg
########################################################################################################################

def test_v2(model, ks, ckg, n_batchs=4):
    ks = eval(ks)
    train_user_dict, test_user_dict = ckg.train_ui_user_dict, ckg.test_ui_user_dict

    n_items = ckg.items_num  # 项目数
    n_test_users = len(test_user_dict)

    n_k = len(ks)
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
    }

    n_users = ckg.users_num
    batch_size = n_users // n_batchs
    for batch_id in tqdm(range(n_batchs + 1), ascii=True, desc="Evaluate"):
        s = batch_size * batch_id
        t = batch_size * (batch_id + 1)
        if t > n_users:
            t = n_users
        if s == t:
            break

        score_matrix = get_score(model, n_users, n_items, train_user_dict, s, t)
        for i, k in enumerate(ks):
            precision, recall, ndcg, hr = 0, 0, 0, 0
            _, topk_index = torch.topk(score_matrix, k)
            topk_index = topk_index.cpu().numpy() + n_users

            for u in range(s, t):
                gt_pos = test_user_dict[u]
                topk = topk_index[u - s]
                num_pos = len(gt_pos)

                topk_set = set(topk)
                test_set = set(gt_pos)
                num_hit = len(topk_set & test_set)

                precision += num_hit / k
                recall += num_hit / num_pos
                hr += 1 if num_hit > 0 else 0

                ndcg += cal_ndcg(topk, test_set, num_pos, k)

            result["precision"][i] += precision / n_test_users
            result["recall"][i] += recall / n_test_users
            result["ndcg"][i] += ndcg / n_test_users
            result["hit_ratio"][i] += hr / n_test_users

    return result
