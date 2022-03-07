#!/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 2020/7/19 11:30 
@Author : Kunsong Zhao
@Version：1.0
"""
import os

__author__ = 'kszhao'


import pandas as pd

color2num = {'r': 1, 'g': 2, 'b': 3}


def convert_num(color):
    return color2num[color]


def convert_path(path):
    return path[:-4]


def convert_file(root_path):

    all_csv = []
    for r, d, f in os.walk(root_path):
        for file in f:
            if ".csv" in file:
                all_csv.append(os.path.join(r, file))

    all_csv = [convert_path(path) for path in all_csv]

    print(all_csv)

    for path in all_csv:

        df = pd.read_csv(path + '.csv')

        columns = df.columns.tolist()

        print(columns)

        values = df.iloc[0, :].values

        values = [convert_num(val) for val in values]
        print(values)

        # 生成csv文件
        csv_df = pd.DataFrame([values], columns=columns)
        csv_df.to_csv(path + '.xlsx' + '.csv', index=False)

        # 生成txt文件
        with open(path + '.xlsx' + '.txt', 'w', encoding='utf-8') as fw:
            fw.write('"x"' + '\n')
            for i in range(len(columns)):
                fw.write('"' + columns[i] + '"' + ' ' + str(values[i]) + '\n')


if __name__ == '__main__':
    convert_file('./scoring/')