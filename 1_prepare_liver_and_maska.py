import nibabel as nib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import Reza_functions2 as rf

dicom_folder="dicom_for_liver_seg/"
mask_folder="mask_for_liver_seg/"
dicom_path=glob.glob(dicom_folder+"*.nii.gz")
mask_path=glob.glob(mask_folder+"*.nii.gz")

output_file="liver_npy/"
if not os.path.isdir(output_file):
    os.mkdir(output_file)
        
def keep_non_zero(index,liver_dicom,liver_mask,around_liver,dicom):
    dicom=dicom.get_data()
    Data_lung  = []
    Maska_train = []
    Data_dicom=[]
    for idx in range(liver_mask.shape[0]):
        if ~( np.sum(np.sum(np.sum(liver_mask[idx,:,:]))) == 0): 
            d=liver_dicom [idx,:,:]
            maska =around_liver [idx,:,:]
            dcm=dicom[idx,:,:]
            
            Data_lung.append(d)
            Maska_train.append(maska)
            Data_dicom.append(dcm)               
    Data_lung  = np.array(Data_lung)
    Maska_train = np.array(Maska_train)
    Data_dicom=np.array(Data_dicom)
    

    for i in range(Data_lung.shape[0]):
        np.save(output_file+'Dicom_lung_' + index + "_" + str(i), Data_lung[i])
        np.save(output_file+'MaskA_' + index + "_" + str(i), Maska_train[i])

    Data_soft = rf.hu_to_grayscale(Data_dicom,400,50)
    for i in range(Data_soft.shape[0]):
        np.save(output_file+'Dicom_soft_' + index + "_" + str(i), Data_soft[i])
    Data_bone = rf.hu_to_grayscale(Data_dicom,1800,400)
    for i in range(Data_bone.shape[0]):
        np.save(output_file+'Dicom_bone_' + index + "_" + str(i), Data_bone[i])
    




if __name__=='__main__':
    for m_path,d_path in zip(mask_path,dicom_path):
        print(m_path)
        mask=nib.load(m_path)
        dicom=nib.load(d_path)
        index=(m_path.replace(".nii.gz","")).split("/")[1]#自己3
        liver_dicom, liver_mask, around_liver, FOV = rf.return_axials(dicom, mask,1500,-600)#rotate=True,flip=False
        keep_non_zero(index,liver_dicom,liver_mask,around_liver,dicom)
        # liver_dicom, liver_mask, around_liver, FOV = rf.return_axials(dicom, mask,400,50)
        # keep_non_zero(index,liver_dicom,liver_mask,around_liver,'soft')
        # liver_dicom, liver_mask, around_liver, FOV = rf.return_axials(dicom, mask,1800,400)
        # keep_non_zero(index,liver_dicom,liver_mask,around_liver,'bone')
    


   
       

