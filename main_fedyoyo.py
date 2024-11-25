from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data_loader.autoaug import *
from models.Resnet8 import ResNet_cifar

import copy
import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from data_loader.tools import *
from data_loader.imbalance_cifar import *
from data_loader.imbalance_svhn import *
from losses import *
EPS = 1e-6
 
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dst', default='cifar10', help='dataset setting')
parser.add_argument('--num_classes', default=10)
parser.add_argument('--arch', metavar='ARCH', default='resnet8')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--noniid', default=0.01, type=float, help='noniid')
parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

# fedavg
parser.add_argument('--method', type=str, default="test")
parser.add_argument('--loss_type', type=str, default='CE')
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--num_online_clients', type=int, default=8)
parser.add_argument('--frac', type=float, default=0.4)
parser.add_argument('--num_rounds', type=int, default=200)
parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
parser.add_argument('--batch_size_local_training', type=int, default=32) #128
parser.add_argument('--batch_size_test', type=int, default=500) #128
parser.add_argument('--lr_local_training', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--tau', default=1.5, type=float)
parser.add_argument('--T', '--temperature', default=1.5, type=float, help='distillation temperature')
parser.add_argument('--lamda', default=1.5, type=float)
parser.add_argument('--warmup', default=50, type=int)
parser.add_argument('--s_epo_la', default=0, type=int)
parser.add_argument('--gamma', type=float, default=0.1)


class Global(object):
    def __init__(self,
                 num_classes: int,
                 args,
                ):
        self.num_classes = num_classes
        self.args = args
        self.criterion = CrossEntropyLoss().to(self.args.device)
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(self.args.device)

        self.test_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).cuda()
      
    def initialize_for_model_fusion(self, list_local_params: list, list_nums_local_data: list):
        total_data = sum(list_nums_local_data)
        fedavg_global_params = {k: torch.zeros_like(v, device=self.args.device) for k, v in list_local_params[0].items()}
        for name_param in fedavg_global_params:
            weighted_sum = sum(
                dict_local_params[name_param].to(self.args.device) * num_local_data
                for dict_local_params, num_local_data in zip(list_local_params, list_nums_local_data)
            )
            fedavg_global_params[name_param] = weighted_sum / total_data
        return fedavg_global_params


    def  global_eval(self, fedavg_params, data_test, batch_size_test,log=None, flag='val'):
        top1 = AverageMeter('Acc@1', ':6.2f')
        self.test_model.load_state_dict(fedavg_params)
        self.test_model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            val_loader=DataLoader(data_test, batch_size_test,shuffle=False)
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda()
                target = target.cuda()
                features, output = self.test_model(input)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], input.size(0))
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            cf = confusion_matrix(all_targets, all_preds).astype(float)

            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('{flag} Results: Prec@1 {top1.avg:.3f}'
                    .format(flag=flag,top1=top1))
            out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
            if log is not None:
                log.write(output + '\n')
                log.write(out_cls_acc + '\n')
                log.flush()
        return top1.avg


    def download_params(self):
        return self.syn_model.state_dict()
      
    def cal_prior(self, eff_global):
        prior = eff_global.clone().detach().to(self.args.device)
        return (prior / prior.sum()).detach()

class Local(object):
    def __init__(self,
                 data_client,
                #  cls_num_list,
                 args
                 ):
        self.args = args       

        self.data_client = data_client
        self.criterion = CrossEntropyLoss().to(self.args.device)

        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).cuda()
        self.teacher_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(self.args.device)
        self.optimizer = SGD(self.local_model.parameters(),lr=args.lr_local_training)

    def local_train(self, args, global_params, r, log_training, cls_num_list, prior=None):
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        local_dis = torch.tensor(cls_num_list, dtype=torch.float, device='cuda')
        local_dis /= local_dis.sum()
        
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True,
                                     num_workers=8,  )
            for data_batch in data_loader:
                _, img1, img2, labels = data_batch 
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()               
                data = torch.cat([img1, img2], dim=0).cuda()
                target = torch.cat([labels, labels], dim=0).cuda()
                
                _, logits = self.local_model(data) 
                prior = (1-args.gamma) * prior + args.gamma * local_dis
                logits = logits + torch.log(torch.pow(prior, args.tau) + 1e-9)
                num = int(target.shape[0] / 2)
                teacher_logits = logits[:num, :]
                student_logtis = logits[num:, :]
                teacher_softmax = F.softmax(teacher_logits / args.T, dim=1).detach()
                student_softmax = F.log_softmax(student_logtis / args.T, dim=1)
                teacher_max, teacher_index = torch.max(F.softmax((teacher_logits), dim=1).detach(), dim=1)
                patitial_target = target[:num]
                kd_loss = F.kl_div(student_softmax[(teacher_index == patitial_target)],
                                teacher_softmax[(teacher_index == patitial_target)],
                                reduction='batchmean') 
                
                kd_loss = 0 if torch.isnan(kd_loss) else kd_loss 
                ce_loss = self.criterion(logits, target)
                loss = ce_loss + args.lamda * kd_loss * min(r / args.warmup, 1.0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        return self.local_model.state_dict()


    def get_feature_mean(self, global_params, cls_num_list):
        self.local_model.load_state_dict(global_params)
        self.local_model.eval()
        cls_num = len(cls_num_list)
        out_dim = self.local_model.classifier.in_features
        feature_mean_end = torch.zeros(cls_num, out_dim).cuda() 
        data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=self.args.batch_size_local_training,
                                 shuffle=True)
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                img0, _, _, labels = data_batch  # 32,3,32,32
                img0, labels = img0.cuda(), labels.cuda()
                features, output = self.local_model(img0)
                features = features.detach()
                for out, label in zip(features, labels):
                    feature_mean_end[label] = feature_mean_end[label] + out

            img_num_list_tensor = torch.tensor(cls_num_list).unsqueeze(1).cuda()
            for i in range(cls_num):
                if cls_num_list[i] > 0: 
                    feature_mean_end[i] = torch.div(feature_mean_end[i], img_num_list_tensor[i]).detach()
        return feature_mean_end

    def calculate_eff_weight(self, train_propertype, cls_num_list):
        self.local_model.eval()
        train_propertype = train_propertype.cuda()
        class_num = len(cls_num_list)
        eff_all = torch.zeros(class_num).float().cuda()
        data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=self.args.batch_size_local_training,
                                 shuffle=True,
                                 )
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                img0, _, _, labels = data_batch  # 32,3,32,32
                data, target = img0.cuda(), labels.cuda()
                features, output = self.local_model(data)
                mu = train_propertype[target].detach()  # batch_size x d
                feature_bz = (features.detach() - mu)  # Centralization
                index = torch.unique(target)  # class subset
                index2 = target  
                eff = torch.zeros(class_num).float().cuda()

                for i in range(len(index)):  # number of class
                    index3 = (index2 == index[i]).nonzero(as_tuple=False).squeeze()  
                    feature_juzhen = feature_bz[index3].detach()  

                    if feature_juzhen.dim() == 1:
                        eff[index[i]] = 1
                    else:
                        
                        _matrixA_matrixB = torch.matmul(feature_juzhen, feature_juzhen.transpose(0, 1))  # n×n
                        _matrixA_norm = torch.unsqueeze(torch.sqrt(torch.sum(feature_juzhen ** 2, axis=1)), 1)  
                        _matrixA_matrixB_length = torch.matmul(_matrixA_norm, _matrixA_norm.transpose(0, 1))  
                        _matrixA_matrixB_length[_matrixA_matrixB_length == 0] = EPS  
                        r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length) 
                        num = feature_juzhen.size(0)  
                        a = (torch.ones(1, num).float().cuda()) / num  # a_T
                        b = (torch.ones(num, 1).float().cuda()) / num  # a
                        c = torch.matmul(torch.matmul(a, r), b).float().cuda()  # a_T R a
                        if c < EPS:
                            c = EPS
                        eff[index[i]] = 1 / c
                eff_all = eff_all + eff

        return eff_all

def FedYoYo():
    args = parser.parse_args()
    args.store_name = '_'.join([args.method,args.dst,'gamma'+str(args.gamma),'warmup'+str(args.warmup), 'lamda'+str(args.lamda), args.arch, 'noniid'+str(args.noniid), 'imb'+str(args.imb_factor)])
    prepare_folders(args)
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    log_eff = open(os.path.join(args.root_log, args.store_name, 'log_effall.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)  
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    # ===================================================
    random_state = np.random.RandomState(args.seed)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])

    augmentation_weak = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    test_trsfm = transforms.Compose([
           # transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])

    if args.dst == 'cifar10':
        # Load data
        augmentation_strong = [
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),  # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        data_local_training = IMBALANCECIFAR10('./data', train=True, download=True, transform=[test_trsfm, transforms.Compose(augmentation_weak),transforms.Compose(augmentation_strong)], imb_type=args.imb_type, imb_factor=args.imb_factor)
        data_global_test = datasets.CIFAR10(root='./data', train=False, transform=test_trsfm)
        args.num_classes = 10

    elif args.dst == 'cifar100':
        # Load data
        augmentation_strong = [
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),  # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        data_local_training = IMBALANCECIFAR100(root='./data', train=True, download=True, transform=[test_trsfm, transforms.Compose(augmentation_weak),transforms.Compose(augmentation_strong)], imb_type=args.imb_type, imb_factor=args.imb_factor)
        data_global_test = datasets.CIFAR100(root='./data', train=False, transform=test_trsfm)
        args.num_classes = 100
    
    elif args.dst == 'svhn':
        # Load data
        augmentation_strong = [
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                SVHNPolicy(),  # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        data_local_training = IMBALANCESVHN(root='./data', split='train', download=True,transform=[test_trsfm, transforms.Compose(augmentation_weak),transforms.Compose(augmentation_strong)], imb_type=args.imb_type, imb_factor=args.imb_factor)
        data_global_test = datasets.SVHN(root='./data', split='test',download=True,  transform=test_trsfm)
        args.num_classes = 10


    # Distribute data
    list_label2indices_train_new = new_classify_label(data_local_training, args.num_classes)
   
    
    if args.noniid == 1:
    
        list_client2indices = iid_clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients,  args.seed) 
    else:
        list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.noniid, args.seed) 

    
    original_dict_per_client = new_show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)

    global_model = Global(num_classes=args.num_classes,
                          args=args,
                          )
    total_clients = list(range(args.num_clients))
    indices2data = new_Indices2Dataset(data_local_training) 
    best_acc1 = 0
    global_acc_list = []
    local_avg_acc_list = []
    eff_global_pre = torch.zeros(args.num_classes).float().cuda()                      
    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        online_clients = list(random_state.choice(total_clients, args.num_online_clients, replace=False))               
        list_dicts_local_params = []
        list_nums_local_data = []
        eff_global_cur = torch.ones(args.num_classes).float().cuda()
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            local_model = Local(data_client=data_client,args=args,)
            cls_num_list=original_dict_per_client[client]           
            # local_model.
            local_feature_mean = local_model.get_feature_mean(global_params,cls_num_list)
            local_feature_mean=local_feature_mean.detach()
            eff_all = local_model.calculate_eff_weight(local_feature_mean,cls_num_list)
            eff_global_cur = eff_global_cur + eff_all
          
        if r > 1 :  
            eff_global_cur = eff_global_pre*0.9 + eff_global_cur*0.1
            eff_global_cur = apply_change_threshold(eff_global_cur, eff_global_pre, 100)
        else:
            abnomal = eff_global_cur > 1e+4
            mean = eff_global_cur[~abnomal].mean() 
            eff_global_cur[abnomal] = mean

        eff_global_pre = eff_global_cur.clone()

        if r % 50 == 0 or r == 1:
            log_eff.write(f'cur epo: {r}, global: {eff_global_cur}'+ '\n')
            log_eff.flush()
        prior = global_model.cal_prior(eff_global_cur)
        
        for client in online_clients:
            # local update
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            cls_num_list=original_dict_per_client[client]
            local_model = Local(data_client=data_client,args=args,)
            local_params = local_model.local_train(args, copy.deepcopy(global_params), r, log_training,cls_num_list,prior=prior)
            list_dicts_local_params.append(copy.deepcopy(local_params)) 
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data) 
       
        acc1 = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
        global_acc_list.append(acc1)
        print('\n'+f"Round {r}, Global Acc: {acc1:.2f}")
        
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_params = copy.deepcopy(fedavg_params)

        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
    
    global_save_path = os.path.join(args.root_log, args.store_name, 'best_model.pth')
    torch.save(best_params, global_save_path)
    global_save_path = os.path.join(args.root_log, args.store_name, 'last_global_model.pth')
    torch.save(copy.deepcopy(fedavg_params), global_save_path)
    print(global_acc_list)     

    log_testing.write(str(global_acc_list) + '\n')
    log_testing.flush()        
    


if __name__ == '__main__':
    FedYoYo()


