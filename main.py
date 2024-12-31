import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
from util.my_dataset import my_dataset, my_dataset_MMSE
from util.calculate_IG import inte_gradient, T_SNE
from model.train_model import train_model, Group_Kfold, Kfold_varity
from model.network_init import getAvgpoolKernelSize, getFlattenDim
from sklearn import decomposition as dp

# 运行环境设置
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 定义参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--decoder_fn', type=str, default='Decoder_mlp')  # 指定解码器函数，默认为MLP解码器
parser.add_argument('--encoder_fn', type=str, default='VariationalEncoder') # 指定编码器函数，默认为变分编码器
parser.add_argument('--out_dir',  type=str, default='./output')  # 输出文件目录
parser.add_argument('--data_dir', type=str, default='./data')  # 数据文件目录
parser.add_argument('--data_filename', type=str, default='MCAD_130_mmse_cut.hdf5')  # 数据文件名
parser.add_argument('--batch_size', default=4, type=int)  # 批大小
parser.add_argument('--lr', default=1e-4, type=float)  # 学习率
parser.add_argument('--total_epochs', default=100, type=int)  # 总训练轮次
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)  # 设备类型
parser.add_argument('--n_fold', default=5, type=int)  # 折数，用于K折交叉验证
parser.add_argument('--group_nfold', default=7, type=int)  # 组折数，用于分组K折交叉验证
parser.add_argument('--n_category', default=2, type=int)  # 分类类别数
parser.add_argument('--n_channal', default=263, type=int)  # 数据通道数
parser.add_argument('--linear_dim', default=0, type=int)  # 全连接层的维度
parser.add_argument('--avgpool_kernelsize', default=43, type=int)  # 平均池化层的核大小
parser.add_argument('--step_size', default=50, type=int)  # 学习率步长
parser.add_argument('--latent_dim', default=10, type=int)  # 潜在空间维度
parser.add_argument('--invmu', default=0.5, type=float)  # 权重平衡参数
parser.add_argument('--rec_kl_scale', default=1e-2, type=float)  # 重建与KL散度的缩放因子
parser.add_argument('--others_shape', default=9, type=int)  # 其他输入数据的形状
args = parser.parse_args()  # 解析参数
# args = parser.parse_args("") # 用于在notebook中测试

# 加载数据
# 使用h5py加载HDF5格式的数据
# 数据文件应包含：X（特征）、y（标签）、site（站点）、sex（性别）、age（年龄）等
data = h5py.File(os.path.join(args.data_dir, args.data_filename), 'r')

# 打印数据的键值
list(data.keys())  # ['X', 'age', 'sex', 'site', 'y'] 
X = data["X"]
y = data["y"]
site = data["site"]
sex = data["sex"]
age = data["age"]
print(site.shape, sex.shape, age.shape)

# 计算其他特征的总维度（若年龄为多维，则求和；否则+1）
if len(age.shape) > 1:
    args.others_shape = site.shape[1] + sex.shape[1] + age.shape[1]
else:
    args.others_shape = site.shape[1] + sex.shape[1] + 1
print(args.others_shape)

# 如果数据文件名包含“mmse”，则加载MMSE评分数据
if 'mmse' in args.data_filename:
    mmse = data["mmse"]

# 打印数据形状信息
print("X: ", X.shape)  # (809, 263, 169)
print("y: ", y.shape)  # <HDF5 dataset "y": shape (809, 1), type "<i4">
print("site: ", site.shape)
print("sex: ", sex.shape)
print("age: ", age.shape)
if 'mmse' in args.data_filename:
    print("mmse:", mmse.shape)

# 对于只分类NC和AD的情况，去除MCI标签的数据
if args.n_category == 2 and (('ADNI' in args.data_filename) or ('MCAD' in args.data_filename)):
    print("Remove the MCI condition, only classify NC and AD.")
    y = y[:]
    ind = np.where(y != 1)  # 去除MCI标签（1）
    print("Num of items kept: ", len(ind[0]))
    X = X[ind]
    y = y[ind]
    site = site[ind]
    sex = sex[ind]
    age = age[ind]
    if 'mmse' in args.data_filename:
        mmse = mmse[ind]
    ind = (y == 2)
    y[ind] = 1  # 将AD标签从2改为1

# 更新参数中的通道数和输入特征数
args.n_channal = X.shape[1]  # 例如MCAD数据集为263，ABIDE数据集为392
args.in_features = X.shape[1] * X.shape[2]  # 通道数 * 每个通道的数据点数

# 构建自定义数据集
# 若包含MMSE数据则使用my_dataset_MMSE，否则使用my_dataset
dataset = my_dataset_MMSE(X, y, site, sex, age, mmse)

# 生成随机数据用于计算模型参数
x_syn = torch.randn((1, X.shape[1], X.shape[2]))

# 获取平均池化层的核大小
args.avgpool_kernelsize = getAvgpoolKernelSize(x_syn, args.n_channal)
print('args.avgpool_kernelsize', args.avgpool_kernelsize)

# 获取全连接层的输入维度
flatten_dim = getFlattenDim(x_syn, args.n_channal, args.avgpool_kernelsize)
print(site[0])
print(sex.shape, site.shape)
args.linear_dim = flatten_dim
print(args.linear_dim)

# 打印最终的参数配置
print(args)

# 使用分组K折交叉验证方法对数据集进行训练
# 具体训练过程由Group_Kfold函数实现
Group_Kfold(dataset, args)
