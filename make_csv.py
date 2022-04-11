from __init__ import *


def get_dict_csv(data_csv_path, data_train_path, save_path):
    train_csv = pd.read_csv(data_csv_path)

    name_list = os.listdir(data_train_path)
    train_csv_all = train_csv.loc[train_csv['image'].isin(name_list), ['image', 'individual_id']]
    train_csv_all.index = range(len(train_csv_all))
    train_csv_all_id = train_csv_all['individual_id'].unique()

    dict_id_all = dict(zip(train_csv_all_id, range(len(train_csv_all_id))))
    info_json = json.dumps(dict_id_all, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(os.path.join(save_path, "dict_id"), 'w')
    f.write(info_json)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_csv_all.to_csv(os.path.join(save_path, "train_csv_train.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成训练所需dict、csv的参数设置')
    parser.add_argument('--data_csv_path', type=str, help='全体训练集csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--data_train_path', type=str, help='训练集路径', required=True)
    args = parser.parse_args()

    get_dict_csv(data_csv_path=args.data_csv_path,
                 data_train_path=args.data_train_path,
                 save_path=args.save_path)
