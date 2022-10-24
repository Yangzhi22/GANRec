"""
# 作者： 杨志
# 时间： 2020年11月19日

"""
import argparse


def parse_args():     # 定义参数设置函数

    parser = argparse.ArgumentParser(description="Parameters of RL-GAN-Recommender.")

    # ------------------------------------------------------- 参数设置 --------------------------------------------------
    parser.add_argument("--data_path", nargs="?", default="./Dataset/", help="数据集的路径")
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="数据集：yelp2018, last-fm, amazon-book")
    parser.add_argument("--emb_size", type=int, default=64, help="嵌入长度")
    parser.add_argument("--regs", nargs="?", default="1e-5", help="用户和项目嵌入的正则化损失")   # 1e-5 for last-fm
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu的id")

    parser.add_argument("--dlr", type=float, default=0.0001, help="决定器的学习率")
    parser.add_argument("--glr", type=float, default=0.0001, help="生成器的学习率")

    # ------------------------------------------------------- 生成器的参数设置 --------------------------------------------
    parser.add_argument("--edge_threshold", type=int, default=64, help="知识图谱过滤边的范围")
    parser.add_argument("--num_sample", type=int, default=32, help="图神经网络采样的数目")
    parser.add_argument("--k_step", type=int, default=1, help="当前正例项目的K步邻接")
    parser.add_argument("--in_channel", type=str, default="[64, 32]", help="图神经网络的输入形状")
    parser.add_argument("--out_channel", type=str, default="[32, 64]", help="图神经网络的输出形状")
    parser.add_argument(
        "--pretrain_g",
        type=bool,
        default=False,
        help="是否加载预训练的生成器"
    )

    # ------------------------------------------------------- 识别器的参数设置 --------------------------------------------
    parser.add_argument("--batch_size", type=int, default=1024, help="训练集的 batch_size")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="测试集的 batch_size")
    parser.add_argument("--num_threads", type=int, default=16, help="程序运行的线程数")
    parser.add_argument("--epoch", type=int, default=1000, help="程序训练的步数")
    parser.add_argument("--show_step", type=int, default=1, help="测试结果的显示步数")
    parser.add_argument("--adj_epoch", type=int, default=1, help="每多少步构建一次邻接矩阵")
    parser.add_argument("--pretrain_d", type=bool, default=False, help="是否加载预训练的识别器")
    parser.add_argument("--freeze_d", type=bool, default=True, help="是否固定识别器的参数")
    parser.add_argument("--freeze_g", type=bool, default=True, help="是否固定生成器的参数")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/best_ab.ckpt",
        help="数据集的预训练结果：best_yelp.ckpt, best_fm.ckpt, best_ab.ckpt",
    )
    parser.add_argument("--out_dir", type=str, default="./weights/", help="模型的输出目录")
    parser.add_argument("--flag_step", type=int, default=64, help="早停的步数设置")
    parser.add_argument("--gamma", type=float, default=0.99, help="奖励积累的参数设置")

    # ----------------------------------------------------- 评估指标的参数设置 --------------------------------------------
    parser.add_argument("--Ks", nargs="?", default="[20, 40, 60, 80, 100]", help="评估指标的参数列表")

    return parser.parse_args()
