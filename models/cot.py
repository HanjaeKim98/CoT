import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ViT
from .basic_layers import MLP, BasicConv
from .word_embedding_utils import initialize_wordembedding_matrix

def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss

class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)

class COT(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(COT, self).__init__()
        self.cfg = cfg
        self.dset = dset
        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs).cuda()

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        self.tot_attrs = torch.LongTensor(list(range(self.num_attrs))).cuda()
        self.tot_objs = torch.LongTensor(list(range(self.num_objs))).cuda()
        self.emb_dim = cfg.MODEL.emb_dim

        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)

        if not cfg.TRAIN.use_precomputed_features and not cfg.TRAIN.comb_features:
            self.feat_extractor = ViT('B_16', pretrained=True)

        feat_dim = cfg.MODEL.img_emb_dim

        obj_emb_modules = [
            nn.Linear(feat_dim, self.emb_dim)
        ]
        attr_emb_modules = [
            nn.Linear(feat_dim*3, self.emb_dim),
        ]

        self.objc_embedder = nn.Sequential(*obj_emb_modules)
        self.attrc_embedder = nn.Sequential(*attr_emb_modules)

        self.img_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp) 
        self.attr_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.obj_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        
        self.conv_3 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)
        self.conv_2 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)
        self.conv_1 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)

        self.mlp_3 = nn.Sequential(
            nn.Linear(self.emb_dim + feat_dim, self.emb_dim + feat_dim // 16),
            nn.ReLU(),
            nn.Linear(self.emb_dim + feat_dim // 16, feat_dim)
            )
        
        self.mlp_2 = nn.Sequential(
            nn.Linear(self.emb_dim + feat_dim, self.emb_dim + feat_dim // 16),
            nn.ReLU(),
            nn.Linear(self.emb_dim + feat_dim // 16, feat_dim)
            )
        
        self.mlp_1 = nn.Sequential(
            nn.Linear(self.emb_dim+768, self.emb_dim+768 // 16),
            nn.ReLU(),
            nn.Linear(self.emb_dim+768 // 16, 768)
            )
        
        self.crossentropy = CrossEntropyLoss()

        self.projection_1 = MLP(
                self.word_dim * 2, self.emb_dim, self.emb_dim, 2, batchnorm=False,
                drop_input=cfg.MODEL.wordemb_compose_dropout
            )

    def compose_visual(self, obj_feats, att_feats):
        inputs = torch.cat([obj_feats, att_feats], 1)
        output = self.projection_1(inputs)
        return output


    def _setup_word_composer(self, dset, cfg):
        attr_wordemb, self.word_dim = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.attrs, cfg)
        obj_wordemb, _ = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.objs, cfg)

        self.attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
        self.obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
        self.attr_embedder.weight.data.copy_(attr_wordemb)
        self.obj_embedder.weight.data.copy_(obj_wordemb)

        self.wordemb_compose = cfg.MODEL.wordemb_compose
        
        self.compose = nn.Sequential(
                nn.Linear(self.word_dim *2 , self.word_dim *3),
                nn.BatchNorm1d(self.word_dim*3),
                nn.ReLU(0.1),
                nn.Linear(self.word_dim *3, self.word_dim * 2),
                nn.BatchNorm1d(self.word_dim*2),
                nn.ReLU(0.1),
                nn.Linear(self.word_dim * 2, self.emb_dim)
            )

    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            attr_emb = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs) # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.attr_embedder(self.all_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.attr_embedder(self.unseen_pair_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.unseen_pair_objs)
        else:
            attr_emb = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs) # # [n_pairs, word_dim].

        concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb)

        return concept_emb



    def train_forward_augment(self, batch, lam=0.5):
        img1 = batch['img']
        img2_o = batch['img1_o'] # Image that shares the same object

        # Labels of 1st image.
        attr_labels_a = batch['attr']
        attr_labels_b = batch['attr1_o']
        obj_labels_a = batch['obj']
        obj_labels_b = batch['obj1_o']
        pair_labels_a = batch['pair']
        pair_labels_b = batch['idx1_o']

        # concat image & label
        img1 = torch.cat((img1, img2_o))
        pair_labels = torch.cat((pair_labels_a, pair_labels_b))
        obj_labels = torch.cat((obj_labels_a, obj_labels_b))
        attr_labels = torch.cat((attr_labels_a, attr_labels_b))


        # generate word embedding
        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs) # 440 300

        bs = img1.shape[0]
        concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)

        
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)


        vis_obj = self.objc_embedder(cls_token)
        obj_pred = self.classifier(vis_obj, obj_weight) 
        
        vis_obj_l9 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )
        
        vis_obj_l6 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )

        vis_obj_l3 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )                

        relation1 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l9).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_3(relation1)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l9)
        relation1_inter = img_feat_l9 * c_scale
        
        relation1_s = torch.cat((torch.mean(vis_obj_l9, 1).unsqueeze(1), torch.mean(relation1_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_3(relation1_s))
        
        img_feat_l9 = img_feat_l9 + relation1_inter * s_scale
        

        relation2 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l6).squeeze()), dim=1)
        c_scale = F.sigmoid(self.mlp_2(relation2)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l6)
        relation2_inter = img_feat_l6 * c_scale
        
        relation2_s = torch.cat((torch.mean(vis_obj_l6, 1).unsqueeze(1), torch.mean(relation2_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_2(relation2_s))
        
        img_feat_l6 = img_feat_l6 + relation2_inter * s_scale
        
        
        relation3 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l3).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_1(relation3)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l3)
        relation3_inter = img_feat_l3 * c_scale
        
        relation3_s = torch.cat((torch.mean(vis_obj_l3, 1).unsqueeze(1), torch.mean(relation3_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_1(relation3_s))
        
        img_feat_l3 = img_feat_l3 + relation3_inter * s_scale

        fused_vis = torch.cat((self.img_avg_pool(img_feat_l9).squeeze(), self.img_avg_pool(img_feat_l6).squeeze(), self.img_avg_pool(img_feat_l3).squeeze()), dim=1)
        attr_emb = self.attrc_embedder(fused_vis)
        attr_pred = self.classifier(attr_emb, attr_weight) 
        vis_comp = self.compose_visual(vis_obj, attr_emb)

        # pure cot
        obj_loss = F.cross_entropy(obj_pred, obj_labels) # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]

        attr_loss = F.cross_entropy(attr_pred, attr_labels) # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]

        comp_pred = self.classifier(vis_comp, concept) # number of object class
        comp_loss = F.cross_entropy(comp_pred, pair_labels) 
        comp_pred = torch.max(comp_pred, dim=1)[1]

        # mma augmentation
        
        lam = np.random.beta(1, 1)

        # mixing attribute vectors
        new_attr = attr_weight[attr_labels_a] * lam + attr_weight[attr_labels_b] * (1-lam) # 64 300
        new_obj = obj_weight[obj_labels_a]

        # virtual ground class
        concept_gt = torch.cat((new_obj, new_attr), dim=-1)
        concept_gt = self.compose(concept_gt) # 64 300

        # mixing visual vectors
        vis_comp_a, vis_comp_b = vis_comp.chunk(2,dim=0)
        vis_mix = lam * vis_comp_a + (1 - lam) * vis_comp_b
        
        # vector normalization
        vis_mix = F.normalize(vis_mix, dim=-1) 
        concept_gt = F.normalize(concept_gt, dim=-1)
        concept_all = F.normalize(concept, dim=-1)

        cpositive = torch.sum(vis_mix * concept_gt, dim=1, keepdim=True)
        cnegative = vis_mix @ concept_all.T
        clogits = torch.cat([cpositive, cnegative], dim=1)
        clabels = torch.zeros(len(clogits), dtype=torch.long, device=cpositive.device)
        mix_loss = F.cross_entropy(clogits / 0.1, clabels)   

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (obj_pred == obj_labels)
        correct_attr = (attr_pred == attr_labels)

        loss = comp_loss + 0.4 * obj_loss + 0.6 * attr_loss + mix_loss
              
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_comp.sum(),float(bs)) 
        }

        return out


    def train_forward(self, batch):
        img1 = batch['img']

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        bs = img1.shape[0]

        concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)


        vis_obj = self.objc_embedder(cls_token)
        obj_pred = self.classifier(vis_obj, obj_weight) 
        
        vis_obj_l9 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )
        
        vis_obj_l6 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )

        vis_obj_l3 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )                


        relation1 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l9).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_3(relation1)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l9)
        relation1_inter = img_feat_l9 * c_scale
        
        relation1_s = torch.cat((torch.mean(vis_obj_l9, 1).unsqueeze(1), torch.mean(relation1_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_3(relation1_s))
        
        img_feat_l9 = img_feat_l9 + relation1_inter * s_scale
        

        relation2 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l6).squeeze()), dim=1)
        c_scale = F.sigmoid(self.mlp_2(relation2)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l6)
        relation2_inter = img_feat_l6 * c_scale
        
        relation2_s = torch.cat((torch.mean(vis_obj_l6, 1).unsqueeze(1), torch.mean(relation2_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_2(relation2_s))
        
        img_feat_l6 = img_feat_l6 + relation2_inter * s_scale
        
        
        relation3 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l3).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_1(relation3)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l3)
        relation3_inter = img_feat_l3 * c_scale
        
        relation3_s = torch.cat((torch.mean(vis_obj_l3, 1).unsqueeze(1), torch.mean(relation3_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_1(relation3_s))
        
        img_feat_l3 = img_feat_l3 + relation3_inter * s_scale

        fused_vis = torch.cat((self.img_avg_pool(img_feat_l9).squeeze(), self.img_avg_pool(img_feat_l6).squeeze(), self.img_avg_pool(img_feat_l3).squeeze()), dim=1)
        attr_emb = self.attrc_embedder(fused_vis)
        attr_pred = self.classifier(attr_emb, attr_weight) 
        vis_comp = self.compose_visual(vis_obj, attr_emb)


        obj_loss = F.cross_entropy(obj_pred, batch['obj']) # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]

        attr_loss = F.cross_entropy(attr_pred, batch['attr']) # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]

        comp_pred = self.classifier(vis_comp, concept)

        comp_loss = F.cross_entropy(comp_pred, pair_labels) 
        comp_pred = torch.max(comp_pred, dim=1)[1]

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (comp_pred == pair_labels)
        correct_attr = (attr_pred == attr_labels)

        loss = comp_loss + 0.4 * obj_loss + 0.6 * attr_loss     
              
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_comp.sum(),float(bs)) 
        }


        return out

    def val_forward(self, batch):
        img = batch['img']

        concept = self.compose_word_embeddings(mode='val') # [n_pairs, emb_dim].
  
        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img)

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)


        vis_obj = self.objc_embedder(cls_token)
        obj_pred = self.classifier(vis_obj, obj_weight) 
        
        vis_obj_l9 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )
        
        vis_obj_l6 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )

        vis_obj_l3 = vis_obj.reshape((vis_obj.shape[0], vis_obj.shape[1], 1, 1)).repeat(
            1, 1, 14, 14
        )                


        relation1 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l9).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_3(relation1)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l9)
        relation1_inter = img_feat_l9 * c_scale
        
        relation1_s = torch.cat((torch.mean(vis_obj_l9, 1).unsqueeze(1), torch.mean(relation1_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_3(relation1_s))
        
        img_feat_l9 = img_feat_l9 + relation1_inter * s_scale
        

        relation2 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l6).squeeze()), dim=1)
        c_scale = F.sigmoid(self.mlp_2(relation2)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l6)
        relation2_inter = img_feat_l6 * c_scale
        
        relation2_s = torch.cat((torch.mean(vis_obj_l6, 1).unsqueeze(1), torch.mean(relation2_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_2(relation2_s))
        
        img_feat_l6 = img_feat_l6 + relation2_inter * s_scale
        
        
        relation3 = torch.cat((vis_obj, self.img_avg_pool(img_feat_l3).squeeze()), dim=1) 
        c_scale = F.sigmoid(self.mlp_1(relation3)).unsqueeze(2).unsqueeze(3).expand_as(img_feat_l3)
        relation3_inter = img_feat_l3 * c_scale
        
        relation3_s = torch.cat((torch.mean(vis_obj_l3, 1).unsqueeze(1), torch.mean(relation3_inter, 1).unsqueeze(1)), dim=1)
        s_scale = F.sigmoid(self.conv_1(relation3_s))
        
        img_feat_l3 = img_feat_l3 + relation3_inter * s_scale


        fused_vis = torch.cat((self.img_avg_pool(img_feat_l9).squeeze(), self.img_avg_pool(img_feat_l6).squeeze(), self.img_avg_pool(img_feat_l3).squeeze()), dim=1)
        attr_emb = self.attrc_embedder(fused_vis)
        vis_comp = self.compose_visual(vis_obj, attr_emb)

        obj_pred = self.classifier(vis_obj, obj_weight, scale=False)
        obj_pred = obj_pred.index_select(1, self.pairs[:, 1])
        attr_pred = self.classifier(attr_emb, attr_weight, scale=False)
        attr_pred = attr_pred.index_select(1, self.pairs[:, 0])

        comp_pred = self.classifier(vis_comp, concept, scale=False)
        
        pred = comp_pred + attr_pred + obj_pred 
        out = {}
        out['pred'] = pred

        out['scores'] = {}
        for _, pair in enumerate(self.val_pairs):
            out['scores'][pair] = pred[:,self.pair2idx[pair]]

        return out
    
    def forward(self, x, flag=False):
        if self.training:
            if flag:
                out = self.train_forward_augment(x)
            else:
                out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out



class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred