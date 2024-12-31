import h5py
import os
from util.load_data import Mcad_ADNI, MCAD_gmv, Mcad_130, ADNI_130
import argparse

# 设置数据目录
DATA_DIR = './MCAD'
datadir = os.path.join(DATA_DIR, "MCAD_BN_Atlas_ts")
data_infofile = os.path.join(DATA_DIR, "mcad_info_809.xlsx")

# 设置命令行参数解析器
# 运行命令：python newdata.py 或带参数运行：python newdata --dataset_name 'new_altas'
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./MCAD', help='数据根目录')
parser.add_argument('--dataset_name', type=str, default='MCAD', help='数据集名称')
parser.add_argument('--option', type=str, default='cut', help='选项')
parser.add_argument('--n_category', default=3, type=int, help='类别数量，仅适用于MCAD数据集')
parser.add_argument('--data_type', default='fixed', type=str, help='数据类型')
args = parser.parse_args()

def save_data():
    # 根据数据集名称加载对应数据
    if args.dataset_name == 'MCAD_ADNI_new':
        X, y, site, sex, age = Mcad_ADNI(args.data_root, args.option)
        fn = args.dataset_name + '_' + args.option + '.hdf5'
    elif args.dataset_name == 'MCAD_gmv':
        X, y, site, sex, age, gmv = MCAD_gmv(args.data_root, args.option)
        fn = 'MCAD_relgm' + '_' + args.option + '.hdf5'
    elif args.dataset_name == 'MCAD_130':
        X, y, site, sex, age, mmse = Mcad_130(args.data_root, args.option)
        fn = 'MCAD_130_mmse' + '_' + args.option + '.hdf5'
    elif args.dataset_name == 'ADNI_130':
        X, y, site, sex, age = ADNI_130(args.data_root, args.option)
        fn = 'ADNI_130_drop' + '_' + args.option + '.hdf5'
    else:
        print("选项错误")
        return

    # 设置输出文件路径
    out_fn = os.path.join(os.path.join(args.data_root, args.dataset_name), fn)
    print("数据保存路径: ", out_fn)
    print("标签数据: ", y)

    # 保存数据到HDF5文件
    f1 = h5py.File(out_fn, "w") 
    f1.create_dataset("X", X.shape, dtype='f', data=X)                 # (809, 263, 229)
    f1.create_dataset("y", y.shape, dtype='i', data=y)                 # 标签数据 (809)
    f1.create_dataset("site", site.shape, dtype='f', data=site)        # 站点数据 (809)
    f1.create_dataset("sex", sex.shape, dtype='f', data=sex)           # 性别数据 (809)
    f1.create_dataset("age", age.shape, dtype='f', data=age)           # 年龄数据 (809)
    if 'gmv' in args.dataset_name:
        f1.create_dataset("gmv", gmv.shape, data=gmv)                  # GMV 数据
    if 'mmse' in out_fn:
        f1.create_dataset("mmse", mmse.shape, data=mmse)               # MMSE 数据
    f1.close()

save_data()

# 旧代码片段（注释掉的示例代码，供参考）
# DATA_DIR = 'E:\\data\\fMRI\\MCAD'
# datadir = os.path.join(DATA_DIR, "MCAD_BN_Atlas_ts")
# data_infofile = os.path.join(DATA_DIR, "mcad_info_809.xlsx")
# 
# # X, y, site, sex, age = load_data_v2(datadir, data_infofile)
# # f1 = h5py.File(os.path.join(datadir,"data_all_pts_1hot.hdf5"), "w")
# 
# # 剪裁数据，仅使用开始的169个点
# X, y, site, sex, age = load_data(datadir, data_infofile)
# f1 = h5py.File(os.path.join(datadir,"data_cut_2_169pts_1hot.hdf5"), "w")
# 
# f1.create_dataset("X", X.shape, dtype='f', data=X)                 # (809, 263, 229)
# f1.create_dataset("y", y.shape, dtype='i', data=y)                 # (809)
# f1.create_dataset("site", site.shape, dtype='f', data=site)        # (809)
# f1.create_dataset("sex", sex.shape, dtype='f', data=sex)           # (809)
# f1.create_dataset("age", age.shape, dtype='f', data=age)           # (809)
# 
# f1.close()
