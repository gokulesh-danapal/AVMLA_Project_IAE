
import tensorflow as tf
import numpy as np

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416
num_classes = 3

class Preprocessor(object):
    def __init__(self,data='combined'):
        self.data = data

    def __call__(self,example):
        features = self.parse(example)
        image = tf.io.parse_tensor(features['image'],out_type = tf.float32)
        velo = tf.io.parse_tensor(features['velo'],out_type = tf.float32)
        classes = tf.sparse.to_dense(features['classes'])
        classes = self.one_hot_encode(classes)
        bboxes = tf.stack([ 
                        tf.sparse.to_dense(features['xmins']),
                        tf.sparse.to_dense(features['ymins']),
                        tf.sparse.to_dense(features['xmaxs']),
                        tf.sparse.to_dense(features['ymaxs'])
                        ],axis=1)
        if self.data == 'combined':
            x = np.concatenate((image,velo),axis=-1)
            x = tf.cast(x, tf.float32)
        elif self.data == 'image':
            x = image
        elif self.data == 'lidar':
            x = velo
        y = (self.preprocess_label_for_one_scale(classes, bboxes, 52,np.array([0, 1, 2])),
             self.preprocess_label_for_one_scale(classes, bboxes, 26,np.array([3, 4, 5])),
             self.preprocess_label_for_one_scale(classes, bboxes, 13,np.array([6, 7, 8])))
        return x , y
    
    def one_hot_encode(self,classes):
        class_list=tf.TensorArray(tf.int32, size=0,dynamic_size=True)
        for j in classes:
            if j == 'Car':
                class_list = class_list.write(class_list.size(),0)
            elif j == 'Cyclist':
                class_list = class_list.write(class_list.size(),1)
            else:
                class_list = class_list.write(class_list.size(),2)
        classes_oh = tf.one_hot(class_list.stack(),num_classes)
        return classes_oh

    def preprocess_label_for_one_scale(self,classes,bboxes,grid_size=13,valid_anchors=None):
        y = tf.zeros((grid_size, grid_size, 3, 5 + num_classes))
        anchor_indices = self.find_best_anchor(bboxes)
        tf.Assert( classes.shape[0]== bboxes.shape[0], [classes])
        tf.Assert(anchor_indices.shape[0] == bboxes.shape[0], [anchor_indices])
        num_boxes = tf.shape(classes)[0]

        indices = tf.TensorArray(tf.int32, num_boxes, dynamic_size=False)
        updates = tf.TensorArray(tf.float32, num_boxes, dynamic_size=False)

        valid_count = 0
        for i in tf.range(num_boxes):
            curr_class = tf.cast(classes[i], tf.float32)
            curr_box = bboxes[i]
            curr_anchor = anchor_indices[i]
            anchor_found = tf.reduce_any(curr_anchor == valid_anchors)
            if anchor_found:
                adjusted_anchor_index = tf.math.floormod(curr_anchor, 3)
                curr_box_xy = (curr_box[..., 0:2] + curr_box[..., 2:4]) / 2
                curr_box_wh = curr_box[..., 2:4] - curr_box[..., 0:2]
                grid_cell_xy = tf.cast(curr_box_xy // tf.cast((1 / grid_size), dtype=tf.float32), tf.int32)
                index = tf.stack([grid_cell_xy[1], grid_cell_xy[0], adjusted_anchor_index])
                update = tf.concat(values=[curr_box_xy, curr_box_wh,tf.constant([1.0]), curr_class],axis=0)
                indices = indices.write(valid_count, index)
                updates = updates.write(valid_count, update)
                valid_count = 1 + valid_count
        y = tf.tensor_scatter_nd_update(y, indices.stack(), updates.stack())
        return y

    def find_best_anchor(self,y_box):
        box_wh = y_box[..., 2:4] - y_box[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, tf.shape(anchors_wh)[0], 1))
        intersection = tf.minimum(box_wh[..., 0],anchors_wh[..., 0]) * tf.minimum(box_wh[..., 1], anchors_wh[..., 1])
        box_area = box_wh[..., 0] * box_wh[..., 1]
        anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.int32)
        return anchor_idx

    def parse(self,example_proto):
        image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'velo': tf.io.FixedLenFeature([], tf.string),
                'classes': tf.io.VarLenFeature(tf.string),
                'xmins': tf.io.VarLenFeature(tf.float32),
                'ymins': tf.io.VarLenFeature(tf.float32),
                'xmaxs': tf.io.VarLenFeature(tf.float32),
                'ymaxs': tf.io.VarLenFeature(tf.float32),
        }
        return tf.io.parse_single_example(example_proto,image_feature_description)