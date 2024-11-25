import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math
import functools
import torch
def apply_change_threshold(cur_eff_all, previous_eff_all, change_threshold):
  
    epsilon = 1e-8  
    change_ratio = torch.abs((cur_eff_all - previous_eff_all) / (previous_eff_all + epsilon))
    exceed_threshold_mask = change_ratio > change_threshold
    

    cur_eff_all[exceed_threshold_mask] = previous_eff_all[exceed_threshold_mask]
    
 
    
    return cur_eff_all.clone()

def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes #max=5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0))) #num=5000,3237,2096
            img_num_per_cls.append(int(num))
    return img_num_per_cls
def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
   
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
 
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)
    
    
    list_clients_indices = [] 
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class] 
        np.random.shuffle(indices) 
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    return img_num_list, list_clients_indices , num_list_clients_indices

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1

def new_classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, (img0, img1, img2, target, index) in enumerate(dataset):
       
        list1[target].append(idx) 

      
    return list1

def new_new_classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, target in enumerate(dataset.target):
        
        list1[target].append(idx)  

        # list1[datum[1]].append(idx)
    return list1

def imagenet_classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for index, target in enumerate(dataset.labels):
       
        list1[target].append(index)  
    return list1

def show_clients_data_distribution(dataset, clients_indices: list, num_classes ):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)] #十个0
        for idx in indices:
            label = dataset[idx][1] 
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client
def new_show_clients_data_distribution(dataset, clients_indices: list, num_classes ):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)] #十个0
        for idx in indices:
            # label = dataset[idx][3] 
            label = dataset.target[idx]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client
def new_imagenet_show_clients_data_distribution(dataset, clients_indices: list, num_classes ):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)] #十个0
        for idx in indices:
            label = dataset.targets[idx] 
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client
class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)
 
class new_Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img0, img1,img2, label, index  = self.dataset[idx]
        return img0, img1,img2, label

    def __len__(self):
        return len(self.indices)

class aug_Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img1,img2, label, index  = self.dataset[idx]
        return img1,img2, label

    def __len__(self):
        return len(self.indices)
class noaug_Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img, label, index  = self.dataset[idx]
        return img, label

    def __len__(self):
        return len(self.indices)
class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose
def get_merge(tail_list,head_list):
    merged_list = []
    for row1, row2 in zip(tail_list, head_list):
        merged_row = row1 + row2
        merged_list.append(merged_row)
    return merged_list
    

def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    indices2targets = [] 
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients) 
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices) 
    list_client2indices = partition_balance(indices_dirichlet, num_clients) 

    return list_client2indices


def iid_clients_indices(list_label2indices: list, num_classes: int, num_clients: int, seed=None):
    
    dic_clients_indices = {}
    classes = list(range(num_classes))
    for _class in classes:
        indices = list_label2indices[_class] 
        np.random.shuffle(indices) 
        index_per_class = split_list_n_list(indices,num_clients)
        for _client in range(num_clients):
            if _class == 0:
                dic_clients_indices[_client] = index_per_class[_client]
            else:
                dic_clients_indices[_client].extend(index_per_class[_client])
    list_clients_indices = list(dic_clients_indices.values())
    return list_clients_indices
def split_list_n_list(origin_list, n):
    result_list=[]
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
 
    for i in range(0, n):
        result_list.append(origin_list[i*cnt:(i+1)*cnt])
    return result_list


def partition_balance(idxs, num_split: int): #idx：len:13996

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split #699,16
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index
  
 
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0] 
                idx_class = _targets[idx_class, 0] 

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch
