# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 03:00:22 2020

@author: GOKI
"""
import numpy as np
import cv2
import glob
import tensorflow as tf

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416
num_classes =3 
output_shape = (416,416)
num_shards=100

img_paths=glob.glob(r"C:\Users\GOKI\Desktop\AI\Datasets\KITTI\dataset\Training\image\*.png")
texts = glob.glob(r"C:\Users\GOKI\Desktop\AI\Datasets\KITTI\dataset\Training\label\*.txt")
lidars = glob.glob(r"C:\Users\GOKI\Desktop\AI\Datasets\KITTI\dataset\Training\velodyne\*.bin")
calibs = glob.glob(r"C:\Users\GOKI\Desktop\AI\Datasets\KITTI\dataset\Training\calib\*.txt") 

def Read_text(text):
   classes=[];xmins=[];xmaxs=[];ymins=[];ymaxs=[]
   f = open(text,'r')
   lines= f.readlines()
   for line in lines:
       if str(line.split(' ')[0]) in ['Car','Cyclist','Pedestrian']:
           classes.append(bytes(line.split(' ')[0],'utf-8'))
           xmins.append(float(line.split(' ')[4])/width)
           ymins.append(float(line.split(' ')[5])/height)
           xmaxs.append(float(line.split(' ')[6])/width)
           ymaxs.append(float(line.split(' ')[7])/height)
     
   return  classes,xmins,ymins,xmaxs,ymaxs
 
def Read_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.preprocessing.image.img_to_array(img)
    return image

def Read_calib(calib):
    calib_data = {}
    with open (calibs[0], 'r') as f :
        for line in f.readlines():
            if ':' in line :
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
        
    Tr_velo_to_cam = np.zeros((4,4))
    Tr_velo_to_cam[3,3] = 1
    Tr_velo_to_cam[:3,:4] = calib_data['Tr_velo_to_cam'].reshape(3,4)
    
    R0_rect = np.zeros ((4,4))
    R0_rect[:3,:3] = calib_data['R0_rect'].reshape(3,3)
    R0_rect[3,3] = 1
    
    P2_rect = calib_data['P2'].reshape(3,4)
    return Tr_velo_to_cam,R0_rect,P2_rect
        
def Read_velo(lidar,width,height,calib):
    velo_int = np.fromfile(lidar, dtype=np.float32).reshape(-1, 4)
    velo = velo_int[:,:3]
    x_int,y_int,z_int,intensity = velo_int[:,0],velo_int[:,1],velo_int[:,2],velo_int[:,3]
    distance = np.sqrt(x_int ** 2 + y_int ** 2 + z_int ** 2)
    
    Tr_velo_to_cam,R0_rect,P2_rect = Read_calib(calib)
    velo_image_coords=[];velo_intensity=[];velo_depth=[]
    for i in range(len(velo)):   
        xnd = np.append(velo[i],1)
        xpnd = Tr_velo_to_cam.dot(xnd)
        xpnd = R0_rect.dot(xpnd)
        xpnd = P2_rect.dot(xpnd)
        w,h =xpnd[0]/xpnd[2],xpnd[1]/xpnd[2]
        if w < width and w > 0 and h < height and h > 0:
            velo_image_coords.append([h,w])
            velo_intensity.append(intensity[i])
            velo_depth.append(distance[i])
    velo_image_coords=np.array(velo_image_coords,dtype=np.int16)
    
    y1=velo_image_coords[:,0]-1
    x1=velo_image_coords[:,1]-1
    velo_array = np.zeros((height,width,2))
    values = [velo_intensity,velo_depth]
    for i in range(len(values)):
        velo_array[y1,x1,i]=values[i]
        img = cv2.normalize(velo_array[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        velo_array[:,:,i] = cv2.bilateralFilter(img,25,150,150)
    return velo_array

def norm_and_combine(velo, image):
    image = image/127.5-1
    velo_image= np.concatenate((image,velo),axis=-1)
    velo_image = tf.cast(velo_image, tf.float32)
    return velo_image
            
def random_flip_image_and_label(image,xmins,xmaxs):
    r = tf.random.uniform([1])
    if r < 0.5:
        image = tf.image.flip_left_right(image)
        #xmin, ymin, xmax, ymax = tf.split(bboxes, [1, 1, 1, 1], -1)
        xmins1,xmaxs1 = [],[]
        for i in range(len(xmins)):
            xmins1.append(1.000-xmaxs[i])
            xmaxs1.append(1.000-xmins[i])
        xmins=xmins1;xmaxs=xmaxs1
        #bboxes = tf.squeeze(
            #tf.stack([xmin, ymin, xmax, ymax], axis=1), axis=-1)
    return image, xmins,xmaxs

def preprocess_label_for_one_scale(classes,bboxes,grid_size=13,valid_anchors=None):
    y = tf.zeros((grid_size, grid_size, 3, 5 + num_classes))
    anchor_indices = find_best_anchor(bboxes)
    tf.Assert(classes.shape[0] == bboxes.shape[0], [classes])
    tf.Assert(anchor_indices.shape[0] == bboxes.shape[0], [anchor_indices])
    num_boxes = tf.shape(classes)[0]

    indices = tf.TensorArray(tf.int32, num_boxes, dynamic_size=False)
    updates = tf.TensorArray(tf.float32, num_boxes, dynamic_size=False)

    valid_count = 0
    for i in tf.range(num_boxes):
        curr_class = tf.cast(classes[i], tf.float32)
        curr_box = bboxes[i]//output_shape[0]
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

def find_best_anchor(y_box):
    box_wh = y_box[..., 2:4] - y_box[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, tf.shape(anchors_wh)[0], 1))
    intersection = tf.minimum(box_wh[..., 0],anchors_wh[..., 0]) * tf.minimum(box_wh[..., 1], anchors_wh[..., 1])
    box_area = box_wh[..., 0] * box_wh[..., 1]
    anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.int32)
    return anchor_idx

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image,velo,xmins,ymins,xmaxs,ymaxs,classes):
    feature = {
        'image':
        _bytes_feature(image),
        'velo':
        _bytes_feature(velo),
        'xmins':
        tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'ymins':
        tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'xmaxs':
        tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'ymaxs':
        tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'classes':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=classes)),
        }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

 
for shard in range(num_shards):
    l_shard=len(texts)/num_shards  
    start = int(shard*l_shard)
    end = int((shard+1)*l_shard)                              
    filepath = r"C:\Users\GOKI\Desktop\AI\Datasets\KITTI\tfrecords\train"+str(shard)+r".tfrecords"
    with tf.io.TFRecordWriter(filepath) as writer:    
        for img_path, text, lidar, calib in zip (img_paths[start:end],texts[start:end],lidars[start:end],calibs[start:end]):
            image = Read_image(img_path)
            height,width = image.shape[0],image.shape[1]
            classes,xmins,ymins,xmaxs,ymaxs= Read_text(text)
            velo = Read_velo(lidar,width,height,calib)
            velo_image = norm_and_combine(velo,image)
            
            velo_image, xmins,xmaxs = random_flip_image_and_label(velo_image,xmins,xmaxs)
                
            velo_image = tf.image.resize(velo_image, output_shape)
            f_image = velo_image[:,:,0:3]
            f_velo = velo_image[:,:,3:5]
            """label = (
                preprocess_label_for_one_scale(classes, bboxes, 52,
                                                    np.array([0, 1, 2])),
                preprocess_label_for_one_scale(classes, bboxes, 26,
                                                    np.array([3, 4, 5])),
                preprocess_label_for_one_scale(classes, bboxes, 13,
                                                    np.array([6, 7, 8])),
            )"""
            bytes_image =  tf.io.serialize_tensor(f_image) 
            bytes_velo = tf.io.serialize_tensor(f_velo)
            example = serialize_example(bytes_image,bytes_velo,xmins,ymins,xmaxs,ymaxs,classes)
            writer.write(example)
    print(shard)