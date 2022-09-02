from __future__ import division
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes, binary_dilation
import matplotlib.pyplot as plt ###
import re
from PIL import Image
from scipy.ndimage import rotate
# Functions
def hu_to_grayscale(volume,WW,WL):
    W_min=WL-WW/2
    W_max=WL+WW/2
    volume = np.clip(volume, W_min, W_max)
    # mxval  = np.max(volume)
    # mnval  = np.min(volume)
    im_volume = (volume - W_min)/max(W_max - W_min, 1e-3)

    return im_volume *255

def get_mask_around(FVO, Mask):
     
    Mask = np.where(Mask>0, 1, 0)
    around_lung = np.where((FVO-Mask)==1, 1, 0)
    return around_lung

def get_mask(segmentation,FOV):
    # initialize output to zeros
    shp    = segmentation.shape
    mask = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    
    # Get mask for kidney and tumor
    mask[np.equal(segmentation,1)] = 255
    mask = np.where((mask+FOV) >255,255,0)
    
    return mask
    
def get_FOV(vol_ims):#around_lung, lung

    volume = np.clip(vol_ims, -512, 512)
    mxval  = np.max(volume)
    mnval  = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)
    # 我的方法
    vol_mask = np.where(im_volume>0, 1, 0) # vol_im 儲存所有有值的部分(人體組織，也就是 不是肺的部分)

    # 對不是費的部分作侵蝕運算，得到中空部分(肺的區域)更廣泛的Mask
    shp = vol_mask.shape
    FOV = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    for idx in range(shp[0]):
        FOV[idx, :, :] = binary_fill_holes(vol_mask[idx, :, :]).astype(vol_mask.dtype)
        FOV[idx, :, :] = binary_erosion(FOV[idx, :, :], structure=np.ones((30,30))).astype(FOV.dtype)
        FOV[idx, :, :] = binary_dilation(FOV[idx, :, :], structure=np.ones((25,25))).astype(FOV.dtype)
    
    return FOV


def return_axials(vol, seg,WW,WL):

    # Prepare segmentation and volume
    vol = vol.get_data()
    seg = seg.get_data()
    # if rotate:#sample=-1#自己則是無s
    #     vol=np.rot90(vol)
    #     seg=np.rot90(seg)
    # if flip:
    #     vol=np.flip(vol,0)
    #     seg=np.flip(seg,0)
    # if vol.shape[0]==512:
    #     vol=np.moveaxis(vol,-1,0)
    #     seg=np.moveaxis(seg,-1,0)
    seg = seg.astype(np.int16)
    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol,WW,WL)
    FOV = get_FOV(vol)
    mask    = get_mask(seg,FOV)
    around_mask = get_mask_around(FOV,mask)

    return vol_ims, mask, around_mask, FOV

def return_axials_augmentation(vol, seg, rotate_num):
    
    # Prepare segmentation and volume
    vol = vol.get_data()
    seg = seg.get_data()
    if vol.shape[0]==512:
        vol=np.moveaxis(vol,-1,0)
        seg=np.moveaxis(seg,-1,0)
    seg = seg.astype(np.int16)

    vol_ro=rotate(vol,rotate_num)
    vol_Img = Image.fromarray(vol_ro)
    vol_resize = vol_Img.resize((vol.shape[0],vol.shape[1],vol.shape[2]), Image.ANTIALIAS)
    vol_np = np.asarray(vol_resize)

    seg_ro=rotate(seg,rotate_num)
    seg_Img = Image.fromarray(seg_ro)
    seg_resize = seg_Img.resize((seg.shape[0],seg.shape[1],seg.shape[2]), Image.ANTIALIAS)
    seg_np = np.asarray(seg_resize)

    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol_np)
    FOV = get_FOV(vol_ims)
    mask    = get_mask(seg_np,FOV)
    around_mask = get_mask_around(FOV,mask)

    return vol_ims, mask, around_mask, FOV

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text): 
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def find_path_size(path):
    #slices = [nib.load(s).get_data() for s in g]
    path_new=[(p.split('_'))[0] for p in path]
    ID=[]
    arc=0
    for i,p in enumerate(path_new):
        if i ==0:
            ID.append(0)
            arc=p
            continue
        if not arc==p:
            ID.append(i-1)
            ID.append(i)
            arc=p
    ID.append(len(path_new)-1) 

    return ID
    
def history_plot(history,weight):

    plt.plot(history.history["loss"],color='r',label='loss')
    plt.plot(history.history["val_loss"],color='g',label='val_loss')
    plt.plot(history.history["lr"],color='y',label='lr')
    plt.legend(loc ='lower right')
    plt.savefig(weight+"hist1.png")
    plt.close()

    plt.plot(history.history["val_binary_io_u"],color='c',label='val_binary_io_u')
    plt.plot(history.history["binary_io_u"],color='b',label='binary_io_u')
    plt.legend(loc ='lower right')
    plt.savefig(weight+"hist2.png")
    plt.close()


