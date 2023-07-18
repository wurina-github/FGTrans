import sys
import os
import argparse
import numpy as np
import shutil

sys.path.append('./')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='change seg to original ply')
    parser.add_argument('--category', type=str, default='bottle')
    args = parser.parse_args()

    data_root = '../datasets'
    split_path = '../DIF/split'
    mode_train = 'train'
    mode_eval = 'eval'

    with open(os.path.join(split_path, mode_train, args.category + '.txt'), 'r') as f:
        file_train = []
        file_train = f.readlines()
        file_train = [i.rstrip().split('/')[0] for i in file_train]
    with open(os.path.join(split_path, mode_eval, args.category + '.txt'), 'r') as f:
        file_eval = []
        file_eval = f.readlines()
        file_eval = [i.rstrip().split('/')[0] for i in file_eval]
    file = file_eval +file_train
    file = np.unique(file)

    for f in os.listdir(os.path.join(data_root,'obj',args.category)):
        if f not in file:
            shutil.rmtree(os.path.join(data_root,'obj',args.category,f))




