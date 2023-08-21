import os
import numpy as np
import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms

from PIL import Image


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
   
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


def imagenet_transform_zappos(phase, cfg):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


class CompositionDataset(tdata.Dataset):
    def __init__(
        self,
        phase,
        split='compositional-split',
        open_world=False,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world

        #num_negs
        self.num_negs = 128

        if 'ut-zap50k' in cfg.DATASET.name:
            self.transform = imagenet_transform_zappos(phase, cfg)
        else:
            self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(f'{cfg.DATASET.root_dir}/images')
        
        self.attrs, self.objs, self.pairs, \
            self.train_pairs, self.val_pairs, \
            self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))


        self.obj_affordance = {} # -> contains objects compatible with an attribute.
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr, obj) in self.train_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

        # Keeping a list of all pairs that occur with each object

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Images that contain an object.
        self.image_with_obj = {}
        self.image_with_obj_hasattr = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
                self.image_with_obj_hasattr[obj] = []
            self.image_with_obj[obj].append(i)
            self.image_with_obj_hasattr[obj].append(self.attr2idx[instance[1]])
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)
        
        unseen_pairs = set()
        for pair in self.val_pairs + self.test_pairs:
            if pair not in self.train_pair2idx:
                unseen_pairs.add(pair)
        self.unseen_pairs = list(unseen_pairs)
        self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}
            
    def get_split_info(self):
        data = torch.load(f'{self.cfg.DATASET.root_dir}/metadata_{self.split}.t7')
        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = \
                instance['image'], instance['attr'], instance['obj'], instance['set']
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if self.cfg.DATASET.name == 'vaw-czsl':
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs
        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/train_pairs.txt')
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/val_pairs.txt')
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/test_pairs.txt')

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        if self.cfg.TRAIN.use_precomputed_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
        unseen = 0
        if (attr, obj) in self.unseen_pairs:
            unseen = 1
        if self.phase == 'train':
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.train_pair2idx[(attr, obj)],
                'unseen': unseen,
                'img_name': self.data[index][0]
            }
            
            # Object task.
            i3 = self.sample_same_object2(attr, obj)
            img1, attr1_o, obj1 = self.data[i3]

            if self.cfg.TRAIN.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)
            data['img1_o'] = img1
            data['attr1_o'] = self.attr2idx[attr1_o]
            data['obj1_o'] = self.obj2idx[obj1]
            data['idx1_o'] = self.train_pair2idx[(attr1_o, obj1)]
            data['img1_name_o'] = self.data[i3][0]
        else:
            # Testing mode.

            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
                'unseen': unseen,
                'name': (attr, obj)
            }
        return data

    def __len__(self):
        return len(self.data)

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_same_object2(self, attr, obj):
        if len(self.obj_affordance[obj]) == 1:
            i2 = np.random.choice(self.image_with_obj[obj])
            return i2
        else:
            i2 = np.random.choice(self.image_with_obj[obj])
            ind, count = np.unique(self.image_with_obj_hasattr[obj], return_counts=True)
            weight = 1.0 / count[ind!=self.attr2idx[attr]]
            weight = weight / np.sum(weight)
            idx = np.random.choice(ind[ind!=self.attr2idx[attr]], p=weight)
            _, attr1, _ = self.data[i2]
            while self.attr2idx[attr1] != idx:
                i2 = np.random.choice(self.image_with_obj[obj])
                _, attr1, _ = self.data[i2]
            return i2


