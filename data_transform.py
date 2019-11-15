#import utility libraries
import os
import sys
import argparse
import glob
import re
import numpy as np 
import cv2
import csv
# using ElementTree to read PascalVOC file
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# supported models
DATA_FORMAT = ["retinanet", "pixellink"]
table_dic = ['textBlock', 'singleText', 'table', 'otherTable', 'stamp', 'title']
#TODO
#define text classes
text_dic = []

def get_annotation_from_CSV(xml_path):
    """Get a list containing all annotations in which row is
        {image_name,width,height,depth,xmin, ymin, xmax, ymax,label}
        
        Args: path of annotation with format of Pascal VOC 
        
        Return: a list containing all annotations
    """
    
    #check if passed path valid
    if not os.path.isfile(xml_path):
        print('Invalid file path: ', xml_path)
        return
    #process xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
#     image_path = root.find('path').text
#     image_path = image_path.replace("\\", "/")
#     dir_start_index = image_path.find(dir_start_chars)
#     image_path = image_path[dir_start_index:]
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    depth =  int(root.find('size').find('depth').text)
    image_name = root.find('filename').text
    
    for obj in root.findall('object'):
        label, xmin, ymin, xmax, ymax = None,None,None,None,None
        label = obj.find('name').text
        for box in obj.findall('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
        annotation.append([image_name,width,height,depth,xmin, ymin, xmax, ymax,label])
    return annotation



def convert_label_for_retinanet(old_class):
    """Convert input class name to new for RetinaNet
    
        Args: old_class: label to be converted
        
        Return converted label name
    """
    
    # labels_in_use = ['TargetTable', 'TargetTablePart', 'NumericTable', 'NumericTablePart', 'AnyTable', 'AnyTablePart']
    labels_in_use = ['textBlock', 'singleText', 'table', 'otherTable', 'stamp', 'title']
    target_table_set = ['TargetTable', 'TargetTablePart']
    numeric_table_set = ['NumericTable', 'NumericTablePart']
    other_table_set = ['AnyTable', 'AnyTablePart']
    
    table_label = 'TargetTable'
    numeric_label = 'NumericTable'
    other_table = 'AnyTable'
    
    if old_class in target_table_set:
        return table_label
    elif old_class in numeric_table_set:
        return numeric_label
    elif old_class in other_table_set:
        return other_table
    else:
        print('Invalid label: {}, please refer to labels in: {}'.format(old_class, labels_in_use))
        return None
    
def convert_label_for_pixellink(old_class):
    """Convert input class name to new for PixelLink
    
        Args: old_class: label to be converted
        
        Return converted label name
    """
#     IMAGE_TYPE_TEXT_DETECTION = ['TopTitleText','TopTitleWord', 'LeftTitleText','LeftTitleWord', 'Text', 'Word']
    IMAGE_TYPE_TEXT_DETECTION = ['TopTitleText','TopTitleWord',  'Text', 'Word']
    top_title_set = ['TopTitleText','TopTitleWord']
#     left_title_set = ['LeftTitleText','LeftTitleWord']
    text_word_set = ['Text', 'Word']
    
    top_title = 'TopTitle'
#     left_title = 'LeftTitle'
    text_word = 'Text'
    
    if old_class in top_title_set:
        return top_title
#     elif old_class in left_title_set:
#         return left_title
    elif old_class in text_word_set:
        return text_word
    else:
        print('\nInvalid label or not used: {}, please refer to labels in: {}'.format(old_class, IMAGE_TYPE_TEXT_DETECTION))
        return None

def generate_RetinaNet(raw_anno_list = None,\
                       save_anno_dir = None,\
                       raw_img_path  = None,\
                       img_save_path =None,\
                       color_ch = None,\
                       output_dim  = (1024, 1024),\
                      detect_type='text_dt'):
    
    anno_counter  = 0
    img_counter   = 0
    num_of_tables = {}
    # padding color
    WHITE = [255,255,255]
    
    content_dic = []
    if detect_type == 'text_dt':
#         content_dic = ['TopTitle', 'LeftTitle', 'Text']
        content_dic = ['TopTitle',  'Text']
    elif detect_type == 'table_dt':
        content_dic = table_dic
        num_of_tables = {ob:0 for ob in table_dic}
    
    # csv file format
    title = ['image_id','xmin','ymin','xmax','ymax','label']
    print('save_anno_dir: {}'.format(save_anno_dir))
    with open(save_anno_dir, "w", encoding="utf-8") as writeFile:
        writer = csv.writer(writeFile)
       # writer.writerow(title)

        for row in raw_anno_list:
            image_name, width, height, depth, xmin, ymin, xmax, ymax,label  = row
            # path of raw image corresponding to current annotation
            img_src = os.path.join(raw_img_path, image_name)
            if os.path.isfile(img_src):  
                # path of new image corresponding to current annotation
                target_file = os.path.join(img_save_path, image_name)
                
                img = cv2.imread(img_src)

                #get resizing ratio
                old_h, old_w = img.shape[:2]
#                 h_resize_ratio, w_resize_ratio = output_dim[0]/height, output_dim[1]/width

                desired_height, desired_width = output_dim[0], output_dim[1]

                # get input image aspect ratio by height/width
                asp_ratio = float(old_h)/float(old_w)
                dst_asp_ratio = float(desired_height)/float(desired_width)

                # get resizing ratio at height and width
                h_ratio, w_ratio = float(desired_height)/old_h, float(desired_width)/old_w
            #     print('h_ratio, w_ratio: {}, {}'.format(h_ratio, w_ratio))

                # direction to which padding will add. 0 for width padding; 1 for height padding
                padding_direction = 0 if asp_ratio > dst_asp_ratio else 1

                #new width and height for resizing
                new_w, new_h = 0, 0
                
                scale = [1, 1]
                # padding toward height direction
                if padding_direction == 1:
                    # resized image will be aligned along width
                    new_w = desired_width
                    new_h = int(new_w*asp_ratio)   
                    ratio = old_w/desired_width
                    scale = [ratio, ratio]
                else: # padding toward width direction
                    # resized image will be aligned along height
                    new_h = desired_height
                    new_w = int(new_h/asp_ratio)
                    ratio = old_h/desired_height
                    scale = [ratio, ratio]

                if color_ch == 'gs':
                    #convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif color_ch == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print('Not currently supported color space: {}'.format(color_ch))
                                
                # resize image to [new_w, new_h]
                resize_img = cv2.resize(img, (new_w, new_h))

                # calculate padding border
                delta_h = desired_height - new_h
                delta_w = desired_width - new_w
                top, bottom, left, right = delta_h//2, delta_h - delta_h//2, delta_w//2,delta_w - delta_w//2
            #     print('top: {}\tbottom: {}\tleft: {}\tright: {}'.format(top, bottom, left, right))
                # add padding to resized image
                padded_im = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=WHITE)
                
#                 if not os.path.isfile(target_file):
#                     # resize to unified size, when resizing parameter is (width, height)
#                     img = cv2.resize(img, tuple(output_dim[::-1]))

                cv2.imwrite(target_file, padded_im)
                print("\nWrite to target file {}".format(target_file))
        
                img_counter +=1
                    
                # add current annotation into csv file     
                class_name = None
                if detect_type == 'text_dt':
                    class_name = convert_label_for_pixellink(label)
                elif detect_type == 'table_dt':
                    # class_name = convert_label_for_retinanet(label)
                    class_name = label
                else:
                    print('Unsupported detection type: {}'.format(detect_type))
                if class_name:
                    xmin = int(int(xmin)/scale[0]) + left
                    ymin = int(int(ymin)/scale[0]) + top
                    xmax = int(int(xmax)/scale[0]) + left
                    ymax = int(int(ymax)/scale[0]) + top
                    
                    if xmax <= xmin:
                        xmax = xmin +1
                    if ymax <= ymin:
                        ymax = ymin + 1
                        
                    writer.writerow([target_file, xmin, ymin, xmax, ymax, class_name])
                    num_of_tables[class_name] += 1
#                    if class_name==content_dic[0]:
#                        num_of_tables[0]+=1
#                    elif class_name == content_dic[1]:
#                        num_of_tables[1]+=1
#                    elif class_name == content_dic[2]:
#                        num_of_tables[2] += 1
                    anno_counter +=1
                else:
                    print('\nBad label conversion: label:{}\tclass name{}'.format(label, class_name))
            else:
                print('\n{} not exists!'.format(img_src))
                
    print("\nWrite to annotation file {}".format(save_anno_dir))
    num_of_label = {}
#    for ty, nm in zip (content_dic, num_of_tables):
#        num_of_label[ty] = nm
        
    return (anno_counter, img_counter, num_of_tables)

def generate_PixelLink(raw_anno_list = None,\
                       save_anno_dir = None,\
                       raw_img_path  = None,\
                       img_save_path =None,\
                       color_ch = None,\
                       output_dim  = (512, 512)):
    
    if not raw_anno_list:
        print('Raw annotation list is empty: {}')
        return
    
    anno_counter  = 0
    img_counter   = 0
    num_of_tables = [0,0,0]
    # loop over all images
    for file in raw_anno_list:
        image_name = file[0][0]
#         print('image_name: {}\t file[0]: {}'.format(image_name, file[0]))
        img_src = os.path.join(raw_img_path, image_name)
        if os.path.isfile(img_src):  
            img = cv2.imread(img_src)
            
            # get resizing ratio
            h, w, _ = img.shape
            h_resize_ratio, w_resize_ratio = output_dim[0]/h, output_dim[1]/w
            
            if color_ch == 'gs':
                #convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif color_ch == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                print('Not currently supported color space: {}'.format(color_ch))
            
            # resize to unified size, when resizing parameter is (width, height)
            img = cv2.resize(img, tuple(output_dim[::-1]))
            # path of new image corresponding to current annotation
            target_file = os.path.join(img_save_path, image_name)
            cv2.imwrite(target_file, img)
            print("\nWrite to target image      => {}".format(target_file))
            img_counter +=1           
        else:
            print('\n{} not exists!'.format(img_src))
            
        gt_save_path = 'gt_' + image_name.split('.')[0] + '.txt'
        gt_save_path = os.path.join(save_anno_dir,gt_save_path)
#         print('gt_save_path: {}'.format(gt_save_path))
#         break
        #loop over all text bboxes of image
        with open(gt_save_path, "w", encoding="utf-8") as textFile:
            
            for row in file:
                image_name,width,height,depth,x0, y0, x1, y1,label  = row
                
                x0, y0, x1, y1 = int(int(x0)*w_resize_ratio),\
                                    int(int(y0)*h_resize_ratio),\
                                    int(int(x1)*w_resize_ratio),\
                                    int(int(y1)*h_resize_ratio)

                # add current annotation into csv file             
                class_name = convert_label_for_pixellink(label)
                if class_name:
                    write_text = str(x0) + ',' + str(y0) +',' + \
                                  str(x1) + ',' + str(y0) + ',' + \
                                  str(x1) + ',' + str(y1) + ',' + \
                                  str(x0) + ',' + str(y1) + ',' + \
                                  label + '\n'
                    textFile.write(write_text)
                    # counting for each class
                    if class_name=='TopTitle':
                        num_of_tables[0]+=1
                    elif class_name == 'LeftTitle':
                        num_of_tables[1]+=1
                    elif class_name == 'Text':
                        num_of_tables[2] += 1 
                    anno_counter +=1
                else:
                    print('\nBad label conversion: label:{}\tclass name{}'.format(label, class_name))
            print("\nWrite to target annotation file => {}".format(gt_save_path))
                
    print("\nWrite to annotation file {}".format(save_anno_dir))
    num_of_label = {}
    for ty, nm in zip (content_dic, num_of_tables):
        num_of_label[ty] = nm
    return (anno_counter, img_counter, num_of_label)


def generate_dataset(dest_data_type =None,\
                     raw_anno_list = None,\
                     save_anno_dir = None,\
                     raw_img_path  = None,\
                     img_save_path =None,\
                     args = None,\
                     resize_ratio  = 1):
    
    """Generate custom annotation csv file and split images into train and validation sets
        Args:
        dest_data_type: data format of destination model
        raw_anno_list: list stored split annotations
        save_anno_dir: file path for saving split annotation file, ex: ../../train.csv
        raw_img_path: path to store original image files
        img_save_path: destination path to store split image data
    """
    
    if not raw_anno_list:
        print('Raw annotation list is empty: {}')
        return
    
    if not save_anno_dir:
        print('Annotation saving path not exists: {}'.format(save_anno_dir))
        return
        
    if not os.path.isdir(raw_img_path):
        print('Raw image path not exists: {}'.format(img_save_path))
        return
        
    if not os.path.isdir(img_save_path):
        print('Image saving path not exists: {}'.format(img_save_path))
        return
    
    color_ch = None
    if args.color_channel:
        color_ch = args.color_channel
    
    output_dim = (2048, 2048)
    if args.output_size:
        try:
            print('Resizing dimension: {}'.format(args.output_size))
            output_dim = np.array(args.output_size)
            if len(output_dim) == 2 and output_dim[0] > 0 and output_dim[1] >0:
                output_dim = args.output_size
            else:
                print('Invalid output dimension {}, image will be saved as (1024, 1024)'.format(output_dim))
                output_dim = (2048, 2048)
        except TypeError as e:
            print('Invalid output dimension {}, image will be saved as (1024, 1024)'.format(output_dim))

    anno_counter  = 0
    img_counter   = 0
    num_of_tables = [0,0,0]
    
    if dest_data_type == DATA_FORMAT[0]:
        anno_counter, img_counter, num_of_tables = generate_RetinaNet(raw_anno_list,\
                                                                      save_anno_dir, \
                                                                      raw_img_path,  \
                                                                      img_save_path, \
                                                                      color_ch,\
                                                                      output_dim,\
                                                                     args.detect_type)
    elif dest_data_type == DATA_FORMAT[1]:
        anno_counter, img_counter, num_of_tables = generate_PixelLink(raw_anno_list,\
                                                                      save_anno_dir, \
                                                                      raw_img_path,  \
                                                                      img_save_path, \
                                                                      color_ch,\
                                                                      output_dim)
    else:
        print('Unsupported model type: {}'.format(dest_data_type))
   
    return (anno_counter, img_counter, num_of_tables)


def parse_args(args):
    """Parse the arguments
    """
    
    parser = argparse.ArgumentParser(prog='transform raw data', description='Transform Pascal VOC data into keras-RetinaNet dataset.')
#     parser.add_argument('--size',type=list, default=(1024,1024), help='Tuple indicating (height, width) to varify')
    parser.add_argument('dest_data_type', default='retinanet', help='Destination file format, currently support: retinanet and pixellink')
    parser.add_argument('--detect_type', default='text_dt', help='Detection type, currently support: text_dt and table_dt')
    parser.add_argument('--anno_path', default='data/Annotations', help='Path to folder/path where annotations locate')
    parser.add_argument('--img_path', default='data/JPEGImages', help='Path to folder/path where images locate')
    #parser.add_argument('--dataset_path', default='data', help='Path to folder/path where train data locate')
    parser.add_argument('--save_path', default='data', help='Path to folder/path where raw data locate')
    parser.add_argument('--train_path', default='train', help='Path to folder/path where train data locate, must be sub-dir of save_path')
    parser.add_argument('--val_path', default='val', help='Path to folder/path where validation data locate, must be sub-dir of save_path')
    parser.add_argument('--test_path', default='test', help='Path to folder/path where test data locate, must be sub-dir of save_path')
    parser.add_argument('--split_val', action='store_true', help='whether split dataset to validation')
    parser.add_argument('--split_test', action='store_true', help='whether split dataset to validation')
    parser.add_argument('--color_channel', default= 'gs', help='Color fomart of output images. Choosing from {gs} for grayscale, or {rgb} for rgb image')
    parser.add_argument('--output_size', nargs='+', type=int, default= (1024, 1024), help='Output image dimension: (height, width)')
    
    return parser.parse_args(args)


def main(args=None):
    
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    dest_data_type = args.dest_data_type.lower()
    if not args.dest_data_type or dest_data_type not in DATA_FORMAT:
        print('Please specify correct destination data format! Current: {}'.format(args.dest_data_type))
    
    if not args.anno_path:
        print('Please specify annotation file path! Current is: {}'.format(args.anno_path))
        return
    anno_path = os.path.abspath(args.anno_path)
    if not os.path.isdir(anno_path):
        print('Annotation file path invalid! Current is: {}'.format(anno_path))
        
        
    if not args.img_path:
        print('Please specify raw image file path! Current is: {}'.format(args.img_path))
        return
    img_path = os.path.abspath(args.img_path)
    if not os.path.isdir(img_path):
        print('Image path invalid! Current is: {}'.format(img_path))
    
    #dataset_path = os.path.abspath(args.dataset_path)
    #if not os.path.isdir(img_path):
     #   print('Destination dataset path invalid! Current is: {}'.format(args.dataset_path))
    
    save_path = os.path.abspath(args.save_path)
    if not os.path.isdir(save_path):
        print('Destination data path not exist!\nCreating: {}'.format(args.save_path))
        try:
            os.mkdir(save_path)
        except FileExistsError as e:
            print(e)
    
    train_path = os.path.join(save_path, args.train_path)
    if not os.path.isdir(train_path):
        print('Destination train path not exist!\nCreating: {}'.format(args.train_path))
        try:
            os.mkdir(train_path)
        except FileExistsError:
            print("Directory is existing!")
            
    if args.split_val:            
        val_path = os.path.join(save_path, args.val_path)
        if not os.path.isdir(val_path):
            print('Destination val path not exist!\nCreating: {}'.format(args.val_path))
            try:
                os.mkdir(val_path)
            except FileExistsError as e:
                print(e)
    
    if args.split_test:
        test_path = os.path.join(save_path, args.test_path)
        if not os.path.isdir(test_path):
            print('Destination train path not exist!\nCreating: {}'.format(args.test_path))
            try:
                os.mkdir(test_path)
            except FileExistsError as e:
                print(e)
        
    # infomation dictionary for debug
    info_dic = []
    # path for train, val and test ground truth        
    train_gt_path = None
    val_gt_path   = None
    test_gt_path  = None
    
    if args.detect_type == 'text_dt':
#         info_dic = ['TopTitle', 'LeftTitle', 'Text']
        info_dic = ['TopTitle', 'Text']
    elif args.detect_type == 'table_dt':
        info_dic = table_dic
    
    print('Detection type: {}\tLabels in use: {}'.format(args.detect_type, info_dic))
    
    if dest_data_type == DATA_FORMAT[0]:        
        train_gt_path = os.path.join(save_path, 'train.csv')
        val_gt_path   = os.path.join(save_path,   'val.csv')
        test_gt_path  = os.path.join(save_path,  'test.csv')
    #  Creating ground truth txt folders for PixelLink       
    elif dest_data_type == DATA_FORMAT[1]:       
        train_gt_path = os.path.join(save_path, 'train_gt')       
        if not os.path.exists(train_gt_path):
            print('\n#############################################################################')
            print("\nCreating train ground truth folders for PixelLink!")
            try:
                os.mkdir(train_gt_path)
            except FileExistsError:
                print("Directory: {} is existing!".format(train_gt_path))
                
        if args.split_val:
            val_gt_path   = os.path.join(save_path, 'val_gt')
            if not os.path.exists(val_gt_path):
                print("\nCreating val ground truth folders for PixelLink!")
                try:
                    os.mkdir(val_gt_path)
                except FileExistsError:
                    print("Directory: {} is existing!".format(val_gt_path)) 
                
        if args.split_test:
            test_gt_path  = os.path.join(save_path, 'test_gt')
            if not os.path.exists(test_gt_path):
                print("\nCreating test ground truth folders for PixelLink!")
                try:
                    os.mkdir(test_gt_path)
                except FileExistsError:
                    print("Directory: {} is existing!".format(test_gt_path))                
    
    # get current path
    #root_dir = os.getcwd()
    anno_dir = os.path.join(anno_path, "*.xml")
    print('\n#############################################################################')
    labels_container = glob.glob(anno_dir) 
    print('\nGet {} annotation fies!'.format(len(labels_container)))
    
    # get all records in annotation files
    print('\n#############################################################################')
    
    img_dir = os.path.join(img_path, '*.jpg')
    print('\n#############################################################################')
    img_container = glob.glob(os.path.join(img_path, '*.jpg'))
    img_container.extend(glob.glob(os.path.join(img_path, '*.png')))
    print('\nGet {} images fies!'.format(len(img_container)))
    
    # get all records in annotation files
    print('\n#############################################################################')  
    dataset_dic = {}
    train_val_test_img = img_container
    # container for annotations
    X_train, X_test, X_val = [], [], []
    # container for images
    X_train_img, X_test_img, X_val_img = [], [], []
    # split data into train and test set
    dataset_dic['train'] = train_val_test_img
    X_train_img = train_val_test_img
    if args.split_test:
        X_train_img, X_test_img = train_test_split(train_val_test_img, test_size = 0.3, random_state=42)
        dataset_dic['train'] = X_train_img
        dataset_dic['test'] = X_test_img

    # split data into train and validation set
    if args.split_val:
        if len(X_test_img) <=0:
            X_train_img, X_val_img = train_test_split(X_train_img, test_size = 0.1, random_state=42)
            dataset_dic['train'] = X_train_img
            dataset_dic['val'] = X_val_img
        else:
            X_test_img, X_val_img = train_test_split(X_test_img, test_size = 0.2, random_state=42)
            dataset_dic['test'] = X_test_img
            dataset_dic['val'] = X_val_img
    
#    FILE_NAME_INDEX = 0
#    CLASS_INDEX = 8
    
    if dest_data_type == DATA_FORMAT[0]:
        print('\nFetch annotations for RetinaNet!')
        for lb, ds in dataset_dic.items():
            for im_pt in ds:
                # get jpg file name
                fln = os.path.split(im_pt)[-1]
                if '.jpg' in fln:
                    b_fln = fln.replace('.jpg', '')
                elif '.png' in fln:
                    b_fln = fln.replace('.png', '')
                a_fln = b_fln + '.xml'
                ann_fpt = os.path.join(anno_path, a_fln)
                if os.path.isfile(ann_fpt):
                    temp_anno = get_annotation_from_CSV(ann_fpt)           
                    for line in temp_anno:
                        if lb == 'train':
                            X_train.append(line)
                        elif lb == 'test':
                            X_test.append(line)
                        elif lb == 'val':
                            X_val.append(line)
                        else:
                            print('bad data structure => {}'.format(lb))
                            return
#         for anno in labels_container:
#             temp_anno = get_annotation_from_CSV(anno)           
#             for line in temp_anno:
# #                 anno_label = line[CLASS_INDEX]
# #                 if anno_label in info_dic:
#                 for lb, ds in dataset_dic.items():                  
#                     train_val_test.append(line)
                    
    elif dest_data_type == DATA_FORMAT[1]:
        print('\nFetch annotations for PixelLink!')
        for anno in labels_container:
            temp_anno = get_annotation_from_CSV(anno)
            train_val_test.append(temp_anno)
            
#     print('\n{} annotated records'.format(len(train_val_test))) 
    
    print('\n#############################################################################')
    print('\nSplit data into:')
    
    print('\ttrain set of size: {}'.format(len(X_train_img)))
    print('\tAnno  set of size: {}'.format(len(X_train)))
    
    if args.split_val:
        print('\tval set of size  : {}'.format(len(X_val_img)))
        print('\tAnno  set of size: {}'.format(len(X_val)))
    
    if args.split_test:
        print('\ttest set of size : {}'.format(len(X_test_img)))
        print('\tAnno  set of size: {}'.format(len(X_test)))

    print('\n#############################################################################')
#     # final annotation file format
#     title = ['image_name','width','height','depth','xmin', 'ymin', 'xmax', 'ymax','label']
#     all_annotations_path = os.path.join(dataset_path, 'raftel_annotations.csv')
#     with open(all_annotations_path, "w", encoding="utf-8") as writeFile:
#         writer = csv.writer(writeFile)
#         writer.writerow(title)
#         for anno in annotation_record_list:
#             writer.writerow(anno)
    
    # get all normal image paths
#     np_raw_labels = np.array(annotation_record_list)
#     train_val_test = []

#     for row in np_raw_labels:
#         image_name, width, height, depth,xmin, ymin, xmax, ymax, class_name = row
#         if width < height:
#             train_val_test.append([image_name,width,height,depth,xmin, ymin, xmax, ymax, class_name ])
    
   
    
    
    train_anno_num, train_img_num, train_num_tables = generate_dataset(dest_data_type, X_train, train_gt_path, img_path, train_path, args)
    
    if args.split_val:
        val_anno_num, val_img_num, val_num_tables   = generate_dataset(dest_data_type, X_val, val_gt_path, img_path, val_path, args)
    
    if args.split_test:
        test_anno_num, test_img_num, test_num_tables= generate_dataset(dest_data_type, X_test, test_gt_path, img_path, test_path, args)
    
    print(\
          '\nGenerated train data:\
          \n\tAnnotations stored at:\n\t{}\
          \n\tImages stored at:\n\t{}\
          \n\tAnnotations number: {}\
          \n\tImage       number: {}\
          \n\tStatistics for each class:'.format(train_gt_path, train_path, train_anno_num, len(X_train_img)))
    
    for lable in info_dic:
        print('\t{}: \t{}'.format(lable, train_num_tables[lable]))

    if args.split_val:
        print(\
              '\nGenerated val data:\
              \n\tAnnotations stored at:\n\t{}\
              \n\tImages stored at:\n\t{}\
              \n\tAnnotations number: {}\
              \n\tImage       number: {}\
              \n\tStatistics for each class:'.format(val_path,\
                                                     val_gt_path, \
                                                     val_anno_num, \
                                                     len(X_val_img)))
        for lable in info_dic:
            print('\t{}: \t{}'.format(lable, val_num_tables[lable]))
              
    if args.split_test:
        print(\
              '\nGenerated test data:\
              \n\tAnnotations stored at:\n\t{}\
              \n\tImages stored at:\n\t{}\
              \n\tAnnotations number: {}\
              \n\tImage       number: {}\
              \n\tStatistics for each class:'.format(test_path, test_gt_path, test_anno_num, len(X_test_img)))
        
        for lable in info_dic:
            print('\t{}: \t{}'.format(lable, test_num_tables[lable]))
                
    print('\n#############################################################################')
    print('Generation completed!')
    
if __name__ == '__main__':
    main()
