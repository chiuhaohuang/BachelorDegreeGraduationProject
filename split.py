import h5py
import numpy as np
import sklearn.preprocessing as pp

# 打开 hdf5 文件
with h5py.File('./data/ADNI_cut.hdf5', 'r') as f:
    # 从文件中读取数据集
    x = f['X'][:]  # 特征数据
    y = f['y'][:]  # 标签数据
    age = f['age'][:]  # 年龄数据
    sex = f['sex'][:]  # 性别数据
    site = f['site'][:]  # 数据采集站点信息
    print(site.shape)  # 打印站点数据的维度
    print(len(y))  # 打印标签数据的数量

    # 按标签值分别统计每一类的数量
    ind1 = y[np.where(y == 0)]  # 标签为 0 的样本
    ind2 = y[np.where(y == 1)]  # 标签为 1 的样本
    inde3 = y[np.where(y == 2)]  # 标签为 2 的样本
    print(len(ind1), len(ind2), len(inde3))  # 输出每类样本的数量
    exit(0)  # 提前退出程序，后续代码不执行（用于调试）

    # 筛选出站点信息为指定值的索引
    target_index = np.where((site == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])).all(axis=1))[0]
    target_index_2 = np.where((site == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])).all(axis=1))[0]
    # 合并两个符合条件的索引数组
    target_index = np.concatenate((target_index, target_index_2))
    
    # 根据筛选出的索引获取对应的数据
    x_target = x[target_index]  # 筛选后的特征数据
    y_target = y[target_index]  # 筛选后的标签数据
    age_target = age[target_index]  # 筛选后的年龄数据
    sex_target = sex[target_index]  # 筛选后的性别数据
    site_target = site[target_index]  # 筛选后的站点数据
    print(site_target.shape)  # 打印筛选后的站点数据维度

# 将筛选后的数据保存到新的 HDF5 文件中
with h5py.File('./data/ADNI_all.hdf5', 'w') as f:
    f.create_dataset('X', data=x_target)  # 保存筛选后的特征数据
    f.create_dataset('y', data=y_target)  # 保存筛选后的标签数据
    f.create_dataset('age', data=age_target)  # 保存筛选后的年龄数据
    f.create_dataset('sex', data=sex_target)  # 保存筛选后的性别数据
    f.create_dataset('site', data=site_target)  # 保存筛选后的站点数据
