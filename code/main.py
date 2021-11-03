import os
import argparse
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from method import Apriori, FPgrowth, gen_association_rules, trans_ibm_data, write_association_rules

parser = argparse.ArgumentParser()
parser.add_argument('--data', default = 'ibm-2021.txt')
parser.add_argument('--out', default = 'output.csv')
parser.add_argument('--func',  default='FPgrowth')
parser.add_argument('--min_support', default = 0.02, type = float)
parser.add_argument('--min_confidence', default = 0.1, type = float)
args = parser.parse_args()

assert args.func in ['FPgrowth', 'Apriori']
print(args)

in_path = '../data/'
out_path = '../result/'

# load data
ibm_data = np.loadtxt(os.path.join(in_path, args.data), dtype=int)
data = trans_ibm_data(ibm_data)

# data preprocess
if args.func == 'FPgrowth':
    data = FPgrowth.preprocess_data(data)
    Model = FPgrowth

elif args.func == 'Apriori':
    te = TransactionEncoder()
    te_data = te.fit(data).transform(data)
    data = pd.DataFrame(te_data, columns=te.columns_).astype(int)
    Model = Apriori

print('Get Frequent Itemsets.')
model = Model(data, args.min_support)
df_freq = model.get_frequent_itemsets()
df_freq['length'] = df_freq.itemsets.apply(len)
df_freq.sort_values(by=['length','support'], ignore_index=True, ascending=[True, False], inplace=True)

print('Gen Association Rules.')
os.makedirs(out_path, exist_ok=True)
df_rule = gen_association_rules(df_freq, args.min_confidence)

print('Write Result.')
write_association_rules(df_rule, path=os.path.join(out_path, args.out))