import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import models_small as M
import numpy as np
import SimpleITK as sitk
import glob
from predict_generater import DataGenerator
from scipy.ndimage.morphology import binary_fill_holes,binary_erosion
from scipy.ndimage import zoom
import Reza_functions2 as rf
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import time
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-w', type=str)
parser.add_argument('-d', type=str)
args = parser.parse_args()

def get_FOV(vol_ims):#around_lung, lung

    vol_mask = np.where(vol_ims>0, 1, 0) # vol_im 儲存所有有值的部分(人體組織，也就是 不是肺的部分)

    # 對不是費的部分作侵蝕運算，得到中空部分(肺的區域)更廣泛的Mask
    shp = vol_mask.shape
    FOV = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    for idx in range(shp[0]):
        FOV[idx, :, :] = binary_fill_holes(vol_mask[idx, :, :]).astype(vol_mask.dtype)
        FOV[idx, :, :] = binary_erosion(FOV[idx, :, :], structure=np.ones((30,30))).astype(FOV.dtype)
        #FOV[idx, :, :] = binary_dilation(FOV[idx, :, :], structure=np.ones((25,25))).astype(FOV.dtype)
    return FOV

def prepare_test(path):
    slice=nib.load(path).get_data()
    

    lung=rf.hu_to_grayscale(slice,1500,-600)
    lung=np.array(lung)

    soft=rf.hu_to_grayscale(slice,400,50)
    soft=np.array(soft)

    bone=rf.hu_to_grayscale(slice,1800,400)
    bone=np.array(bone)

    return lung,soft,bone


def load_niigz(path):
    
    slice_resize=[]
    slice=nib.load(path).get_data()
    for s in slice:
        test_Img = Image.fromarray((s).astype(np.uint8))
        test_small_Img = test_Img.resize((256,256), Image.ANTIALIAS)
        test_data = np.asarray(test_small_Img)
        slice_resize.append(test_data)
    
    slice_resize=np.asarray(slice_resize)
    

    return slice_resize
####################################  Load Data #####################################
print('Get Dataset Path...')

# Parameters
params = {'dim': (256,256),
          'batch_size': 12,
          'n_classes': 1, # 這個好像要處理一下，有看到說可以用1，或4的倍數
          'n_channels': 3,
          'shuffle':False
          }

start=time.time()
dicom_folder = glob.glob('liver_test/*.nii.gz')
dicom_folder.sort(key=rf.natural_keys)
output_path=args.d
weight=args.w
model = M.unet(input_size = (256,256,3))
model.load_weights(weight) 

if not os.path.isdir(output_path):
    os.mkdir(output_path)

done=0
for dicom_path in dicom_folder:
    print(done,'/',len(dicom_folder)-1)
    done+=1
    print(dicom_path)
    dicom_p=nib.load(dicom_path)

    dicom=load_niigz(dicom_path)#標準化
    # FOV=get_FOV(dicom)

    lung,soft,bone=prepare_test(dicom_path)

  
    predict_generator = DataGenerator(lung,soft,bone,**params)

    predictions = model.predict(predict_generator, verbose=1)
    
    predictions=np.squeeze(predictions)
    
    predictions_mask = np.where(predictions>0.5, 1, 0)
    predictions_mask = zoom(predictions_mask, (1,2,2))

    FOV=get_FOV(predictions_mask)

    Estimated_mask = np.where(FOV - predictions_mask>0, 1, 0)##


    Estimated_mask=np.int32(Estimated_mask)

   
 
    out = nib.Nifti1Image(Estimated_mask,dicom_p.affine,dicom_p.header)
    nib.save(out,output_path + 'mask_' + dicom_path.split('/')[-1])
prosses=time.time()-start
print("time:",prosses)









    







