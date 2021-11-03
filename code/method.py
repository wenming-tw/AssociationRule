import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import chain, combinations

class Apriori():
    def __init__(self, data, min_support):
        self.data = data.values.copy()
        self.scale = data.shape[0]
        self.min_support = min_support
        self.item2idx = {item:i for i, item in enumerate(data.columns)}

    def __mining_frequent_1_itemset(self):
        freq_itemsets = dict()
        support = self.data.sum(axis=0) / self.scale
        for i, itemset in enumerate(self.item2idx.keys()):
            if support[i] >= self.min_support:
                freq_itemsets[frozenset([itemset])] = support[i]

        return freq_itemsets

    def __mining_frequent_2_itemset(self, old_frequent_itemsets):
        freq_itemsets = dict()
        candidates = list(combinations(map(lambda x:list(x)[0], old_frequent_itemsets), 2))
        if len(candidates) > 0:
            freq_itemsets = self.__mining(candidates)

        return freq_itemsets

    def __mining_frequent_k_itemset(self, old_frequent_itemsets, k):
        freq_itemsets = dict()
        candidates = list(combinations((set(chain(*old_frequent_itemsets))), k))
        candidates = [c for c in candidates if self.__check_candidate(c, old_frequent_itemsets)]
        if len(candidates) > 0:
            freq_itemsets = self.__mining(candidates)

        return freq_itemsets

    def __mining(self, candidate):
        k = len(candidate[0])
        freq_itemsets = dict()        
        for itemset in candidate:
            idx = [self.item2idx.get(item) for item in itemset]
            support = (self.data[:, idx].sum(axis=1)==k).sum() / self.scale

            if support >= self.min_support:
                freq_itemsets[frozenset(itemset)] = support
        
        return freq_itemsets

    def __check_candidate(self, candidate, old_frequent_itemsets):
        subsets = list(map(set, combinations(candidate, len(candidate)-1)))
        for subset in subsets:
            if subset not in old_frequent_itemsets:
                return False

        return True

    def get_frequent_itemsets(self, max_length=10):
        k = 0
        freq_itemsets = dict()
        if k < max_length:
            now_frequent_itemsets = self.__mining_frequent_1_itemset()
            freq_itemsets.update(now_frequent_itemsets)
            k += 1
        
        if k < max_length:
            old_frequent_itemsets = list(now_frequent_itemsets.keys())
            if len(old_frequent_itemsets) > 0:
                now_frequent_itemsets = self.__mining_frequent_2_itemset(old_frequent_itemsets)
                freq_itemsets.update(now_frequent_itemsets)
                k += 1

        while(k < max_length):
            old_frequent_itemsets = list(now_frequent_itemsets.keys())
            if len(old_frequent_itemsets) > 0:
                now_frequent_itemsets = self.__mining_frequent_k_itemset(old_frequent_itemsets, k+1)
                freq_itemsets.update(now_frequent_itemsets)
                k += 1
            else:
                break
        
        if len(freq_itemsets) == 0:
            return pd.DataFrame(columns=['support', 'itemsets'])

        df_ret = pd.DataFrame(
            np.flip(list(freq_itemsets.items()), axis=1),
            columns=['support', 'itemsets']
        )
        df_ret.support = df_ret.support.astype(np.float64)

        return df_ret

class Node():
    def __init__(self, item, support, parent):
        self.item = item
        self.support = support
        self.parent = parent
        self.child = dict()
        self.next = None

    def increase(self, support=1):
        self.support += support

    def set_next(self, node):
        self.next = node

    def __repr__(self):
        return "<{:}:{:}>".format(self.item, self.support)

    def display(self, ind=0, f=None):
        out = "{}<{:}:{:}>".format('---|'*ind, self.item, self.support)
        if f is not None:
            f.write(out+'\n')            
        else:
            print(out)

        for child in self.child.values():
            child.display(ind+1, f)

class FPgrowth():
    def __init__(self, data, min_support):
        self.data = data
        self.min_support = min_support
        self.root = Node('root', '', None)
        self.__init_htable()
        self.__build_tree()

    def __init_htable(self):
        htable = dict()
        for itemsets, support in self.data.items():
            for item in itemsets:
                htable[item] = htable.get(item, 0) + support

        htable = {item: [cnt, None] for item, cnt in htable.items() if cnt >= self.min_support} # {'item': (support, headerLink)}
        self.htable = OrderedDict(sorted(htable.items(), key=lambda x:(x[1][0], x[0]), reverse=True)) if len(htable) > 0 else None

    def __build_tree(self):
        if self.htable is None:
            return 
            
        for itemsets, support in self.data.items():
            itemsets = [item for item in itemsets if item in self.htable] # remove low support itemset
            itemsets = sorted(itemsets, key=lambda x:(self.htable[x][0], x), reverse=True) # sorted by (support, itemname)

            if len(itemsets) > 0:
                self.__insert(self.root, itemsets, support)

    def __insert(self, node, itemsets, support):
        item = itemsets[0]
        if item in node.child:
            node.child.get(item).increase(support)
        else:
            node.child[item] = Node(item, support, node)
            self.__update_htable(item, node.child.get(item))

        if len(itemsets) > 1:
            self.__insert(node.child.get(item), itemsets[1::], support)
        
    def __update_htable(self, item, node):
        n_node = self.htable.get(item)[1]
        if n_node is None:
            self.htable.get(item)[1] = node
        else:
            while n_node.next is not None:
                n_node = n_node.next
            n_node.set_next(node)

    def __prefix(self, node):
        itemset = list()
        while node.parent is not None:
            itemset.append(node.item)
            node = node.parent
        return itemset

    def prefix_patterns(self, item):
        patterns = dict()
        if self.htable is None:
            return patterns

        source = self.htable.get(item)[1]
        while source is not None:
            pattern = self.__prefix(source)[1:]
            if len(pattern) > 0:
                patterns[frozenset(pattern)] = source.support
            source = source.next
        
        return patterns

    def __mining(self, tree, prefix, frequent_itemsets):
        for item, (support, header) in tree.htable.items():
            now_frequent_itemset = prefix.copy()
            now_frequent_itemset.add(item)

            frequent_itemsets[frozenset(now_frequent_itemset)] = support
            condition_patterns = tree.prefix_patterns(item)
            pattern_tree = FPgrowth(condition_patterns, self.min_support)
            
            if pattern_tree.htable is not None:
                self.__mining(pattern_tree, now_frequent_itemset, frequent_itemsets)

    def get_frequent_itemsets(self):
        if self.htable is None:
            return pd.DataFrame(columns=['support', 'itemsets'])

        freq_itemsets = dict()
        self.__mining(self, set([]), freq_itemsets)

        df_ret = pd.DataFrame(
            np.flip(list(freq_itemsets.items()), axis=1),
            columns=['support', 'itemsets']
        )
        df_ret.support = df_ret.support.astype(np.float64)
        
        return df_ret

    def display(self, path=None):
        if path is not None:
            file = open(path, 'w')
            self.root.display(f=file)
            file.close()
        else:
            self.root.display()

    @staticmethod
    def preprocess_data(data):
        # input: [[...], [...], ...[...]]
        # return: {'...':count, '...':count, ..., '...':count}      
        result = dict()
        for row in data:
            key = tuple(row)
            result[key] = result.get(key, 0) + 1
        
        result = {itemset:count/len(data) for itemset, count in result.items()}
        
        return result

def gen_association_rules(df, min_confidence):
    item2sup = {row[1]:row[0] for row in df.values}
    ret = list()
    for support, itemsets, length in df.values:
        for i in range(1, length):
            for sub_itemset in map(frozenset, combinations(itemsets, i)):
                antecedent_support = item2sup[sub_itemset]
                consequent_support = item2sup[itemsets-sub_itemset]
                confidence = support / antecedent_support
                if confidence >= min_confidence:
                    lift = confidence / consequent_support
                    leverage = support - (antecedent_support*consequent_support)
                    conviction = (1 - consequent_support) / (1-confidence + 1e-9)
                    ret.append([sub_itemset, itemsets-sub_itemset, antecedent_support, consequent_support, support, confidence, lift, leverage, conviction])

    if len(ret) == 0:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction'])

    df_ret = pd.DataFrame(np.array(ret), columns=['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction'])
    return df_ret

def trans_ibm_data(data):
    ndata = list()
    for uid in np.unique(data[:, 0]):
        u_mask = data[:, 0] == uid
        u_data = data[u_mask]
        for tid in np.unique(u_data[:, 1]):
            t_mask = u_data[:, 1] == tid
            ut_data = u_data[t_mask]
            ndata.append(ut_data[:, -1].tolist())
    return  ndata

def write_association_rules(df, path='output.csv'):
    with open(path, 'w') as file:
        file.write(', '.join(["Relationship", "Support", "Confidence", "Lift", "Leverage", "Conviction"]) + '\n')
        for i, row in df.iterrows():
            Relationship = '"{}\' -> \'{}"'.format(set(row.antecedents), set(row.consequents))
            Support = '{:.6f}'.format(row.support)
            Confidence = '{:.6f}'.format(row.confidence)
            Lift = '{:.6f}'.format(row.lift)
            Leverage = '{:.6f}'.format(row.leverage)
            Conviction = '{:.6f}'.format(row.conviction)
            file.write(', '.join([Relationship, Support, Confidence, Lift, Leverage, Conviction]) + '\n')