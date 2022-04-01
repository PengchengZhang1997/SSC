import torch
import torch.nn as nn
from modules.scl import SCL
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        """ Decorrelated learning """
        self.l1 = nn.Linear(self.FeatureExtraction_output, self.FeatureExtraction_output // 2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.FeatureExtraction_output // 2, self.FeatureExtraction_output)
        self.sigmoid = nn.Sigmoid()
        self.l3 = nn.Linear(self.SequenceModeling_output, 4)
        self.l4 = nn.Linear(self.SequenceModeling_output, 4)
        self.soft = nn.Softmax(dim=-1)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.SequenceModeling_style = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

        """ Projection """
        self.project = nn.Sequential(nn.ReLU(), nn.Linear(opt.hidden_size, 128))

        self.scl = SCL()

    def forward(self, input, text, is_train=True, is_domain=False):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        bb = self.sigmoid(self.l2(self.relu(self.l1(torch.mean(visual_feature, 1)))))

        cc = torch.ones(bb.size()[0], bb.size()[1]).cuda()
        dd = cc - bb

        bb = bb.unsqueeze(1)
        dd = dd.unsqueeze(1)

        visual_feature_new = visual_feature * bb

        contextual_feature_new = self.SequenceModeling(visual_feature_new)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature_new.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length, is_domain=is_domain)
        
        if is_domain:
            return prediction, visual_feature, self.Prediction.context_history
        
        elif is_train:
            visual_feature_style = visual_feature * dd
            style_feature = self.SequenceModeling_style(visual_feature_style)

            ee = self.l4(torch.mean(contextual_feature_new, 1))
            ee_softmax = self.soft(ee)
            ee_log = self.logsoft(ee)

            style_ff = self.l3(torch.mean(style_feature, 1))

            mid_len = visual_feature.shape[1] // 2
            visual_feature_aug = torch.cat((visual_feature[:, mid_len:, ], visual_feature[:, :mid_len, ]), dim=1)

            bb_aug = self.sigmoid(self.l2(self.relu(self.l1(torch.mean(visual_feature_aug, 1)))))

            dd_aug = cc - bb_aug

            bb_aug = bb_aug.unsqueeze(1)
            dd_aug = dd_aug.unsqueeze(1)

            visual_feature_new_aug = visual_feature_aug * bb_aug
            contextual_feature_new_aug = self.SequenceModeling(visual_feature_new_aug)

            visual_feature_style_aug = visual_feature_aug * dd_aug
            style_feature_aug = self.SequenceModeling_style(visual_feature_style_aug)

            ee_aug = self.l4(torch.mean(contextual_feature_new_aug, 1))
            ee_softmax_aug = self.soft(ee_aug)
            ee_log_aug = self.logsoft(ee_aug)

            style_ff_aug = self.l3(torch.mean(style_feature_aug, 1))

            style_feature_pro = self.project(style_feature)
            style_feature_aug_pro = self.project(style_feature_aug)
            contextual_feature_new_pro = self.project(contextual_feature_new)
            contextual_feature_new_aug_pro = self.project(contextual_feature_new_aug)

            seq_style = torch.mean(style_feature_pro, 1)
            other_fea = torch.stack([style_feature_aug_pro, contextual_feature_new_pro, contextual_feature_new_aug_pro], 1)
            scloss = self.scl(seq_style, other_fea, 0.1)

            seq_style_aug = torch.mean(style_feature_aug_pro, 1)
            other_fea_aug = torch.stack([style_feature_pro, contextual_feature_new_pro, contextual_feature_new_aug_pro], 1)
            scloss_aug= self.scl(seq_style_aug, other_fea_aug, 0.1)

            return prediction, ee_softmax, ee_log, style_ff, ee_softmax_aug, ee_log_aug, style_ff_aug, scloss, scloss_aug

        else:
            return prediction