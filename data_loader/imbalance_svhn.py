import torchvision
import numpy as np
import torch
from PIL import Image

class IMBALANCESVHN(torchvision.datasets.SVHN):
    cls_num = 10  

    def __init__(self, root, split='train', imb_type='exp', imb_factor=0.01, rand_number=0,
                 transform=None, target_transform=None, download=False, reverse=False):
        super(IMBALANCESVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        np.random.seed(rand_number)
        # img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        # self.gen_imbalanced_data(img_num_list)
        # img_num_list = self.get_cls_num_list()
        # print(img_num_list)
        # self.reverse = reverse
        # self.class_dict = self._get_class_dict()
        self.labels = np.array(self.labels)  
        self.original_counts = np.array([np.sum(self.labels == i) for i in range(self.cls_num)])  
        img_num_list = self.get_img_num_per_cls(self.original_counts, imb_type, imb_factor, reverse)
        print("Desired sample numbers per class:", img_num_list)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    # def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
    #     img_max = len(self.data) / cls_num
    #     img_num_per_cls = []
    #     if imb_type == 'exp':
    #         for cls_idx in range(cls_num):
    #             if reverse:
    #                 num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
    #                 img_num_per_cls.append(int(num))
    #             else:
    #                 num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
    #                 img_num_per_cls.append(int(num))
    #     elif imb_type == 'step':
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max))
    #         for cls_idx in range(cls_num // 2, cls_num):
    #             img_num_per_cls.append(int(img_max * imb_factor))
    #     else:
    #         img_num_per_cls.extend([int(img_max)] * cls_num)
    #     return img_num_per_cls
    def get_img_num_per_cls(self, original_counts, imb_type, imb_factor, reverse):
        sorted_indices = np.argsort(original_counts)[::-1]  
        sorted_counts = original_counts[sorted_indices] 
        
        img_num_per_cls = []
        img_max = len(self.data) / self.cls_num  
        
        for cls_idx in range(self.cls_num):
            if imb_type == 'exp':
                if reverse:
                    num = img_max * (imb_factor**((self.cls_num - 1 - cls_idx) / (self.cls_num - 1.0)))
                else:
                    num = img_max * (imb_factor**(cls_idx / (self.cls_num - 1.0)))
            else:
                num = img_max  
            
           
            img_num_per_cls.append((sorted_indices[cls_idx], int(num)))

     
        return [x[1] for x in sorted(img_num_per_cls, key=lambda x: x[0])]

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64) 
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * len(selec_idx))
            self.num_per_cls_dict[the_class] = len(selec_idx) 
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))  
        
        if self.transform is not None:
            img0 = self.transform[0](img)
            img1 = self.transform[1](img)
            img2 = self.transform[2](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, balanced_data['sample_image'], balanced_data['sample_label'], index
        return img0, img1, img2, target, index
       

    def get_annotations(self):
        annos = []
        for target in self.labels:
            annos.append({'category_id': int(target)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno['category_id']
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict
import torchvision
import numpy as np
import torch
from PIL import Image

class noaug_IMBALANCESVHN(torchvision.datasets.SVHN):
    cls_num = 10  

    def __init__(self, root, split='train', imb_type='exp', imb_factor=0.01, rand_number=0,
                 transform=None, target_transform=None, download=False, reverse=False):
        super(noaug_IMBALANCESVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        np.random.seed(rand_number)
        # img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        # self.gen_imbalanced_data(img_num_list)
        # img_num_list = self.get_cls_num_list()
        # print(img_num_list)
        # self.reverse = reverse
        # self.class_dict = self._get_class_dict()
        self.labels = np.array(self.labels)  
        self.original_counts = np.array([np.sum(self.labels == i) for i in range(self.cls_num)])  
        self.img_num_list = self.get_img_num_per_cls(self.original_counts, imb_type, imb_factor, reverse)
        print("Desired sample numbers per class:", self.img_num_list)
        self.gen_imbalanced_data(self.img_num_list )
        self.reverse = reverse

    # def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
    #     img_max = len(self.data) / cls_num
    #     img_num_per_cls = []
    #     if imb_type == 'exp':
    #         for cls_idx in range(cls_num):
    #             if reverse:
    #                 num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
    #                 img_num_per_cls.append(int(num))
    #             else:
    #                 num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
    #                 img_num_per_cls.append(int(num))
    #     elif imb_type == 'step':
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max))
    #         for cls_idx in range(cls_num // 2, cls_num):
    #             img_num_per_cls.append(int(img_max * imb_factor))
    #     else:
    #         img_num_per_cls.extend([int(img_max)] * cls_num)
    #     return img_num_per_cls
    def get_img_num_per_cls(self, original_counts, imb_type, imb_factor, reverse):
        sorted_indices = np.argsort(original_counts)[::-1] 
        sorted_counts = original_counts[sorted_indices]  
        
        img_num_per_cls = []
        img_max = len(self.data) / self.cls_num  
        
        for cls_idx in range(self.cls_num):
            if imb_type == 'exp':
                if reverse:
                    num = img_max * (imb_factor**((self.cls_num - 1 - cls_idx) / (self.cls_num - 1.0)))
                else:
                    num = img_max * (imb_factor**(cls_idx / (self.cls_num - 1.0)))
            else:
                num = img_max  
            
          
            img_num_per_cls.append((sorted_indices[cls_idx], int(num)))

       
        return [x[1] for x in sorted(img_num_per_cls, key=lambda x: x[0])]

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)  
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * len(selec_idx))
            self.num_per_cls_dict[the_class] = len(selec_idx) 
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))) 
        
        if self.transform is not None:
            img = self.transform(img)
           

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, balanced_data['sample_image'], balanced_data['sample_label'], index
        return img, index
       

    def get_annotations(self):
        annos = []
        for target in self.labels:
            annos.append({'category_id': int(target)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno['category_id']
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict
