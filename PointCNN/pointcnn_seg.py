from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
from pointcnn import PointCNN_SEG


class Net(PointCNN_SEG):
    def __init__(self, points, features, is_training, setting):
        PointCNN_SEG.__init__(self, points, features, is_training, setting)

        ###Classification
        self.classification_logits = pf.dense(self.fc_layers_classification[-1], setting.num_class, 'logits_classification',
                               is_training, with_bn=False, activation=None)

        ###Segmentation Mask
        self.segmentation_logits = pf.dense(self.fc_layers_segmentation[-1], 2, 'logits_segmentation',
                               is_training, with_bn=False, activation=None)