import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from utils import xywh_to_x1x2y1y2, broadcast_iou, binary_cross_entropy

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416


def DarknetConv(inputs, filters, kernel_size, strides, name):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        name=name + '_conv2d',
        use_bias=False,
    )(inputs)
    
    x = BatchNormalization(name=name + '_bn')(x)
    x = LeakyReLU(alpha=0.1, name=name + '_leakyrelu')(x)
    return x


def DarknetResidual(inputs, filters1, filters2, name):
    shortcut = inputs
    x = DarknetConv(
        inputs, filters=filters1, kernel_size=1, strides=1, name=name + '_1x1')
    x = DarknetConv(
        x, filters=filters2, kernel_size=3, strides=1, name=name + '_3x3')
    x = Add(name=name + '_add')([shortcut, x])
    return x


def Darknet(shape=(256, 256, 3)):
    
    inputs = Input(shape=shape)

    x = DarknetConv(inputs, 32, kernel_size=3, strides=1, name='conv2d_0')

    x = DarknetConv(x, 64, kernel_size=3, strides=2, name='conv2d_1')
    # 1x residual blocks
    for i in range(1):
        x = DarknetResidual(x, 32, 64, 'residual_0_' + str(i))

    x = DarknetConv(x, 128, kernel_size=3, strides=2, name='conv2d_2')
    # 2x residual blocks
    for i in range(2):
        x = DarknetResidual(x, 64, 128, 'residual_1_' + str(i))

    x = DarknetConv(x, 256, kernel_size=3, strides=2, name='conv2d_3')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 128, 256, 'residual_2_' + str(i))

    y0 = x

    x = DarknetConv(x, 512, kernel_size=3, strides=2, name='conv2d_4')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 256, 512, 'residual_3_' + str(i))

    y1 = x

    x = DarknetConv(x, 1024, kernel_size=3, strides=2, name='conv2d_5')
    # 4x residual blocks
    for i in range(4):
        x = DarknetResidual(x, 512, 1024, 'residual_4_' + str(i))

    y2 = x

    return tf.keras.Model(inputs, (y0, y1, y2), name='darknet_53')


def YoloV3(shape=(416, 416, 3), num_classes=2, training=False):
    # YoloV3:
    # Yolo predicts 3 boxes at each scale so
    #  the tensor is N × N × [3 ∗ (4 + 1 + C)] for the 4 bounding box offsets,
    #  1 objectness prediction, and C class predictions."
    # 3 * (4 + 1 + num_classes) = 21
    final_filters = 3 * (4 + 1 + num_classes)

    inputs = Input(shape=shape)

    backbone = Darknet(shape)
    x_small, x_medium, x_large = backbone(inputs)

    x = DarknetConv(
        x_large,
        512,
        kernel_size=1,
        strides=1,
        name='detector_scale_large_1x1_1')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_1')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_2')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_2')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_3')

    y_large = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_3')
    y_large = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_large_final_conv2d',
    )(y_large)

    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_0')

    x = UpSampling2D(size=(2, 2), name='detector_scale_1_upsampling')(x)
    x = Concatenate(name='detector_scale_1_concat')([x, x_medium])

    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_1')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_1')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_2')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_2')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_3')

    y_medium = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_3')
    y_medium = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_medium_final_conv2d',
    )(y_medium)

    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_0')
    x = UpSampling2D(size=(2, 2), name='detector_scale_small_upsampling')(x)
    x = Concatenate(name='detector_scale_small_concat')([x, x_small])

    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_1')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_1')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_2')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_2')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_3')

    y_small = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_3')
    y_small = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_small_final_conv2d',
    )(y_small)
    
    y_small_shape = tf.shape(y_small)
    y_medium_shape = tf.shape(y_medium)
    y_large_shape = tf.shape(y_large)

    y_small = tf.reshape(
        y_small, (y_small_shape[0], y_small_shape[1], y_small_shape[2], 3, -1),
        name='detector_reshape_small')
    y_medium = tf.reshape(
        y_medium,
        (y_medium_shape[0], y_medium_shape[1], y_medium_shape[2], 3, -1),
        name='detector_reshape_meidum')
    y_large = tf.reshape(
        y_large, (y_large_shape[0], y_large_shape[1], y_large_shape[2], 3, -1),
        name='detector_reshape_large')

    if training:
        return tf.keras.Model(inputs, (y_small, y_medium, y_large))

    box_small = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[0:3], num_classes),
        name='detector_final_box_small')(y_small)
    box_medium = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[3:6], num_classes),
        name='detector_final_box_medium')(y_medium)
    box_large = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[6:9], num_classes),
        name='detector_final_box_large')(y_large)

    outputs = (box_small, box_medium, box_large)
    return tf.keras.Model(inputs, outputs)


def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_classes):

    t_xy, t_wh, objectness, classes = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1)

    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)

    grid_size = tf.shape(y_pred)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.stack(C_xy, axis=-1)
    C_xy = tf.expand_dims(C_xy, axis=2) 
    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)
    b_xy = b_xy / tf.cast(grid_size, tf.float32)
    b_wh = tf.exp(t_wh) * valid_anchors_wh

    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box, objectness, classes


def get_relative_yolo_box(y_true, valid_anchors_wh):

    grid_size = tf.shape(y_true)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.expand_dims(tf.stack(C_xy, axis=-1), axis=2)

    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]
    t_xy = b_xy * tf.cast(grid_size, tf.float32) - tf.cast(C_xy, tf.float32)

    t_wh = tf.math.log(b_wh / valid_anchors_wh)
    t_wh = tf.where(
        tf.logical_or(tf.math.is_inf(t_wh), tf.math.is_nan(t_wh)),
        tf.zeros_like(t_wh), t_wh)

    y_box = tf.concat([t_xy, t_wh], axis=-1)
    return y_box


class YoloLoss(object):
    def __init__(self, num_classes, valid_anchors_wh):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5

    def __call__(self, y_true, y_pred):
        """
        calculate the loss of model prediction for one scale
        """
        pred_xy_rel = tf.sigmoid(y_pred[..., 0:2])
        pred_wh_rel = y_pred[..., 2:4]
        pred_box_abs, pred_obj, pred_class = get_absolute_yolo_box(
            y_pred, self.valid_anchors_wh, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)
        true_xy_abs, true_wh_abs, true_obj, true_class = tf.split(
            y_true, (2, 2, 1, self.num_classes), axis=-1)
        true_box_abs = tf.concat([true_xy_abs, true_wh_abs], axis=-1)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)


        true_box_rel = get_relative_yolo_box(y_true, self.valid_anchors_wh)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]


        xy_loss = self.calc_xy_loss(true_obj, true_xy_rel, pred_xy_rel, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_rel, pred_wh_rel, weight)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)

        ignore_mask = self.calc_ignore_mask(true_obj, true_box_abs,
                                            pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        return xy_loss + wh_loss + class_loss + obj_loss, (xy_loss, wh_loss,
                                                           class_loss,
                                                           obj_loss)

    def calc_ignore_mask(self, true_obj, true_box, pred_box):
       
        true_box_shape = tf.shape(true_box)
        pred_box_shape = tf.shape(pred_box)
        true_box = tf.reshape(true_box, [true_box_shape[0], -1, 4])
        true_box = tf.sort(true_box, axis=1, direction="DESCENDING")
        true_box = true_box[:, 0:100, :]
        pred_box = tf.reshape(pred_box, [pred_box_shape[0], -1, 4])
        
        iou = broadcast_iou(pred_box, true_box)
        best_iou = tf.reduce_max(iou, axis=-1)
        best_iou = tf.reshape(best_iou, [pred_box_shape[0], pred_box_shape[1], pred_box_shape[2], pred_box_shape[3]])
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):

        obj_entropy = binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

        return obj_loss + noobj_loss

    def calc_class_loss(self, true_obj, true_class, pred_class):
 
        class_loss = binary_cross_entropy(pred_class, true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        return class_loss

    def calc_xy_loss(self, true_obj, true_xy, pred_xy, weight):
 
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        xy_loss = true_obj * xy_loss * weight

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

        return xy_loss

    def calc_wh_loss(self, true_obj, true_wh, pred_wh, weight):

        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
        return wh_loss
