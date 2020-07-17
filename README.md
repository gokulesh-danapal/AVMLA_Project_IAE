# AVMLA_Project_IAE
Project repository for Autonomous vehicles by Machine Learning Algorithms course

## **Project topic : Real time object detection through sensor fusion of camera and lidar**

The code is summarised as follows:
### YoloV3_Kitti.ipynb :
  Jupyter notebook that was used for running these codes on google colab. It also contains code that writes text files based on the predictions of the YOLO network. This step is necessary to evaluate the model by calculating its mean Average Precision

### yoloV3.py:
  Contains the yolo network implemented in tensorflow using keras API. It also contains the custom yolo loss function as defined by the creators. The network accepts inputs in the form of {None X 416 X 416 X no. of channels}. It outputs a three detections at small,medium and large scales. Each scale in turn has three tensors of the form { None X 13 X 13 X 3 X 8 } 
  where:
  
        - 13 is the grid size for small scale detection. It is 26 and 52 for medium and large scale detections respectively.
        - 3 is the number of anchors. In yolo boxes are detected as offsets pre-configured boxes called anchors.
        - 8 is (4 +1+3) that is the Left, top, right and bottom coordinates of the box, confidence and class probabilities
        
   The other two tensors are of the form { 1 X 13 X 13 X 3 X 1 } and { 1 X 13 X 13 X 3 X C }
    where 
    
          - 1 is the objectness score
          - C is the number of classes
### write_tf.py
  It writes tf records from dataset images, lidar points and annotations data. Additionally it also resizes images to (416 X 416) which is the input shape for yolo.
  
### preprocess.py:
  It reads files from tf records file and randomly flips images, lidar and annotations together. The annotations are also resized into yolo output format. This is necessary while training the model and also while validating the model for calculation of various metrics.
          
### train.py: 
   contains the training script. It is designed to avoid plateau-ing of learning rate by dividing learning rate by 10 when a set patience count is reached. Patience count is the number of continuous epochs which have validation loss higher than the latest available lowest validation loss. 

### evaluate.py:
  calculates the mAP (mean Average precision) based on PASCAL 2012 evaluation methodology. Returns the Average precision across classes , PR curve and lamr.
   
### utils.py:
  contains some miscellaneous functions to convert boxes from xmin,ymin,xmax,ymax to x,y,w,h format. A function to calculate iou between sets of boxes. 
 
