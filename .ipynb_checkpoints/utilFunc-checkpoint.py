#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import json
import matplotlib.pyplot as plt
#from datasets import data as dataset


# In[ ]:





# In[81]:


"""
img_Folder_Path_Dir = './Dataset/train2014/train2014'
 
img_List = os.listdir(img_Folder_Path_Dir)
#imgList => 폴더 내 이미지 파일 리스트. 사실 필요 없긴 함.

anno_Path_Dir = './Dataset/annotations_trainval2014/annotations/instances_train2014.json'
with open(anno_Path_Dir, 'r') as f:

    json_Data = json.load(f)

#print("imgList 크기 : " + str(len(imgList)))
#print("jsonData[\"images\"] 크기 : " + str(len(jsonData["images"])))
#print("jsonData[\"annotations\"] 크기 : " + str(len(jsonData["annotations"])))


#print("--------------------image json 값 형태 예시--------------------")
#print("image data json 값 1번 : ", jsonData["images"][0])
#print("image data json 값 2번 : ", jsonData["images"][1])
#print(jsonData["categories"])

#print("--------------------annotation json 값 형태 예시--------------------")
#print("annotation data json 값 1번 : ", jsonData["annotations"][0])
#print("annotation data json 값 2번 : ", jsonData["annotations"][1])

json_Data2 = dict()

for i in range(len(json_Data["annotations"])):
    img_Id = '%012d'%json_Data["annotations"][i]["image_id"]
    ctg_Id = json_Data["annotations"][i]["category_id"]
    bbox   = json_Data["annotations"][i]["bbox"]
    if "COCO_train2014_"+img_Id in json_Data2.keys():
        json_Data2["COCO_train2014_"+img_Id].append([ctg_Id, bbox])
    else:
        json_Data2["COCO_train2014_"+img_Id] = [[ctg_Id, bbox]]
        
#print("-------------------json_data2--------------------")
#print("형태 => ", '파일명 : [[bbox 값 append]]' )

#print("-------------------값 확인(COCO_train2014_000000480023.jpg)-------------------")
#for i in range (len(jsonData2["COCO_train2014_000000480023"])):
    #category = jsonData2["COCO_train2014_000000480023"][i][0]
    #print(json_data2["COCO_train2014_000000480023"][i])

"""


# In[85]:

def sum_of_channel(data):
    batch_size = np.shape(data)[0]
    channel_length = np.shape(data)[1]
    heatmap_width = np.shape(data)[2]
    heatmap_height = np.shape(data)[3]
    
    transposed_output = np.transpose(data, [0, 2, 3, 1])
    tmp = np.zeros((batch_size, heatmap_width, heatmap_height))
    for i in range(batch_size):
        for j in range(channel_length):
            tmp[i] = tmp[i] + data[i, :, :, j]
            
    

    return tmp


def img_resize(img_data, resize_width, resize_height):
    batch_size = len(img_data)
    #img_buffer = np.array([])
    img_buffer = []
    for i in range(batch_size):
        _img_data = np.array(cv2.resize(img_data[i], dsize=(resize_width, resize_height), interpolation=cv2.INTER_LINEAR))
        
        img_buffer.append(_img_data)
        
    return img_buffer

def load_img(img_paths, normalize):
    #make heatmap
    batch_size = len(img_paths)
    img_data = []
    for batch_idx in range(batch_size):
        img = plt.imread(img_paths[batch_idx])
        if(img.ndim==2):
            _img = np.stack((img,)*3, axis=-1)
        else:
            _img = img
            
        if(normalize == True):
            _img = cv2.normalize(_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
        img_data.append(_img)
    
    return img_data

def make_heatmap(img_paths, batch_Anno_Data, heatmap_width, heatmap_height, num_of_class):
    #make heatmap field
    batch_size = len(img_paths)
    heatmap = np.zeros((batch_size, num_of_class, heatmap_height, heatmap_width), np.float32)
    gaussian_heatmap = np.zeros((batch_size, num_of_class, heatmap_height, heatmap_width), np.float32)
    #make heatmap
    for batch_idx in range(batch_size):
        num_of_object = len(batch_Anno_Data[batch_idx])
        img_data = plt.imread(img_paths[batch_idx])
        for anno_idx in range(num_of_object):
            #batch_Anno_Data[batch_idx] => [20, [12.07, 189.09, 57.75, 53.92]]
            object_class = batch_Anno_Data[batch_idx][anno_idx][0]
            object_box   = batch_Anno_Data[batch_idx][anno_idx][1]
            
            img_width  = np.shape(img_data)[1]
            img_height = np.shape(img_data)[0]
            
            center_x = int((object_box[0] + object_box[2]//2) / img_width * heatmap_width)
            center_y = int((object_box[1] + object_box[3]//2) / img_height * heatmap_height)

            heatmap[batch_idx][object_class][center_y][center_x] = 1.0
            #print(object_class)
            gaussian_heatmap[batch_idx][object_class] = cv2.GaussianBlur(heatmap[batch_idx][object_class], (3, 3), 0)
    return heatmap, gaussian_heatmap


def COCO_Object_Detection_2014_TrainData_Load(img_Folder_Path_Dir, anno_Path_Dir):
    #imgFolderPathDir = './Dataset/train2014/train2014'
    train_Folder_Name = img_Folder_Path_Dir
 
    img_List = os.listdir(img_Folder_Path_Dir)
    #imgList => 폴더 내 이미지 파일 리스트. 사실 필요 없긴 함.

    #annoPathDir = './Dataset/annotations_trainval2014/annotations/instances_train2014.json'
    with open(anno_Path_Dir, 'r') as f:

        json_Data = json.load(f)

    json_Data2 = dict()

    for i in range(len(json_Data["annotations"])):
        img_Id = '%012d'%json_Data["annotations"][i]["image_id"]
        ctg_Id = json_Data["annotations"][i]["category_id"]
        bbox   = json_Data["annotations"][i]["bbox"]
        if img_Folder_Path_Dir+"/COCO_train2014_"+img_Id + ".jpg" in json_Data2.keys():
            json_Data2[img_Folder_Path_Dir + "/COCO_train2014_"+img_Id + ".jpg"].append([ctg_Id, bbox])
        else:
            json_Data2[img_Folder_Path_Dir + "/COCO_train2014_"+img_Id + ".jpg"] = [[ctg_Id, bbox]]

    return json_Data2


# In[182]:


#jData2 = COCO_Object_Detection_2014_TrainData_Load('./Dataset/train2014/train2014',
#'./Dataset/annotations_trainval2014/annotations/instances_train2014.json')

def read_Categories_Data_From_COCO_File(path):
    #'./Dataset/annotations_trainval2014/annotations/instances_train2014.json'
    anno_Path_Dir = path
    with open(anno_Path_Dir, 'r') as f:

        json_Data = json.load(f)
    return json_Data["categories"]


#categories_data = read_Categories_Data_From_2014_COCO_File("./Dataset/annotations_trainval2014/annotations/instances_train2014.json")


# In[172]:


def make_Batch(anno_data, batch_size = 16):
    num_of_data = len(anno_data)
    index = np.arange(0, num_of_data)
    np.random.shuffle(index)
    index = index[:batch_size]
    keys = list(anno_data.keys())
    #shuffled_img_data = [img_path[i] for i in index]
    #shuffled_anno_data = [anno_data[j] for j in index]
    shuffled_img_path = [keys[j] for j in index]
    shuffled_anno_data = [anno_data[keys[j]] for j in index]
    #shuffled_anno_data = [anno_data[j:j+1][0][0][0] for j in index]

    
    return shuffled_img_path, shuffled_anno_data


#shuffled_img_path, shuffled_anno_data = make_Batch(jData2, 16)


# In[ ]:





# In[ ]:





# In[ ]:




