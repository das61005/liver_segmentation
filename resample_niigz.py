import nibabel as nib
import numpy as np
import glob
import os
dicom_path='7_21_1/'
mask_path='7_21_2/'

dicom_all=glob.glob(dicom_path+'*')
mask_all=glob.glob(mask_path+'*')

dicom_out_path='dicom_for_liver_seg/'
mask_out_path='mask_for_liver_seg/'
if not os.path.isdir(dicom_out_path):
    os.mkdir(dicom_out_path)
if not os.path.isdir(mask_out_path):
    os.mkdir(mask_out_path)

for dicom_p,mask_p in zip(dicom_all,mask_all):
    print(dicom_p)
    dicom_n=nib.load(dicom_p)
    dicom=dicom_n.get_data()
    mask_n=nib.load(mask_p)
    mask=mask_n.get_data()

    #########################(r,f) sample -1,0;自己的1,無
    dicom=np.rot90(dicom,1) 
    #dicom=np.flip(dicom,0)
    dicom=np.moveaxis(dicom,-1,0) 
    mask=np.rot90(mask,1) 
    #mask=np.flip(mask,0) 
    mask=np.moveaxis(mask,-1,0)
    #########################

    out = nib.Nifti1Image(dicom,dicom_n.affine,dicom_n.header)
    nib.save(out,dicom_out_path + dicom_p.split('/')[-1])
    out = nib.Nifti1Image(mask,mask_n.affine,mask_n.header)
    nib.save(out,mask_out_path +  mask_p.split('/')[-1])


