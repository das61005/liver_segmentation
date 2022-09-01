import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import models_small as M
import numpy as np
import matplotlib.pyplot as plt
import glob
from predict_generater import DataGenerator
from scipy.ndimage.morphology import binary_fill_holes
from PIL import Image
import Reza_functions2 as rf
import nibabel as nib
import argparse
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('-w', type=str)
parser.add_argument('-d', type=str)
args = parser.parse_args()

def get_FOV(mask):#around_lung, lung
    
    shp = mask.shape
    FOV = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    for idx in range(shp[0]):
        FOV[idx, :, :] = binary_fill_holes(mask[idx, :, :]).astype(mask.dtype)
        
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

def save_as_plot(dicom,mask,predictions,output_path):

    predictions = np.squeeze(predictions) # 刪除長度為1的軸
    predictions_mask = np.where(predictions>0.5, 1, 0)
    FOV=get_FOV(predictions)
    Estimated_mask = np.where((FOV - predictions_mask)>0, 1, 0)##


    shp = predictions_mask.shape

    #原圖上遮罩
    dicom_with_mask=np.zeros((shp[0], shp[1] ,shp[2]), dtype=np.float32)

    for i,per_dicom in enumerate(dicom):
        dicom_with_mask[i]=np.where(Estimated_mask[i]==1,255,per_dicom)##
    #若輸入格式改了這個也要改


    print('save_plot')
    for n in range(dicom_with_mask.shape[0]):
        
        plt.subplot(221)
        plt.title("original_mask")
        plt.imshow(mask[n])
        plt.subplot(222)
        plt.title("Estimated_mask")
        plt.imshow(Estimated_mask[n])
        plt.subplot(223)
        plt.title("predict_with_dicom")
        plt.imshow(dicom_with_mask[n])
        plt.subplot(224)
        plt.title('dicom')
        plt.imshow(dicom[n])
        
        plt.savefig(output_path+"/"+str(n)+".png")#
        plt.close()

def save_as_niigz(predictions,dicom_path,output_path):

    predictions = np.squeeze(predictions) # 刪除長度為1的軸
    predictions_mask = np.where(predictions>0.5, 1, 0)

    predictions_mask = zoom(predictions_mask, (1,2,2))

    FOV=get_FOV(predictions_mask)

    Estimated_mask = np.where(FOV - predictions_mask>0, 1, 0)##
    Estimated_mask=np.int32(Estimated_mask)

    dicom_p=nib.load(dicom_path)
    out = nib.Nifti1Image(Estimated_mask,dicom_p.affine,dicom_p.header)
    nib.save(out,output_path + '/mask_' + dicom_path.split('/')[-1])

####################################  Load Data #####################################
print('Get Dataset Path...')

# Parameters
params = {'dim': (256,256),
          'batch_size': 12,
          'n_classes':  1, # 這個好像要處理一下，有看到說可以用1，或4的倍數
          'n_channels': 3,
          'shuffle':False
          }

dicom_folder = 'dicom_for_liver_seg/'
mask_folder = 'mask_for_liver_seg/'

dicom_list=glob.glob(dicom_folder+'*')
mask_list=glob.glob(mask_folder+'*')
dicom_list.sort(key=rf.natural_keys)
mask_list.sort(key=rf.natural_keys)


for dicom_path,mask_path in zip(dicom_list,mask_list):

    print("preparenpy data to list")
    lung,soft,bone=prepare_test(dicom_path)
    mask=load_niigz(mask_path)
    dicom=load_niigz(dicom_path)
    
    output_path=args.d
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path=output_path+dicom_path.split('/')[-1].replace('.nii.gz','')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)


    print("set generator")
    predict_generator = DataGenerator(lung,soft,bone,**params)

    print('Start build model...')
    weight=args.w
    model = M.unet(input_size = (256,256,3))

    print("load weight")
    model.load_weights(weight) 

    print('Start predictions...')
    predictions = model.predict(predict_generator, verbose=1)

    print("save prediction")
    save_as_plot(dicom,mask,predictions,output_path)

    print('save_to_nii.gz')
    save_as_niigz(predictions,dicom_path,output_path)



