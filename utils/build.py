from torch.utils.data import DataLoader
from utils.data_loader import TrainGenerator, TestGenerator


def build_dataset_loader(args_config, graph):
    train_generator = TrainGenerator(args_config=args_config, graph=graph)             # 训练集生成器
    train_loader = DataLoader(                                                         # 训练集加载器
        train_generator,                                                               # 需要一个类
        batch_size=args_config.batch_size,
        shuffle=True,
        num_workers=args_config.num_threads,
    )

    test_generator = TestGenerator(args_config=args_config, graph=graph)              # 测试集生成器
    test_loader = DataLoader(                                                         # 测试集生成器
        test_generator,
        batch_size=args_config.batch_size,
        shuffle=False,
        num_workers=args_config.num_threads,
    )

    return train_loader, test_loader