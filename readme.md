
### 使用不同window(soft tissue,lung,bone)組成三個通道的dicom製作2.5D模型，預測liver segmentation.
![image](https://github.com/das61005/liver_segmentation/blob/main/img/liver1.png)
![image](https://github.com/das61005/liver_segmentation/blob/main/img/hist.png)
## 訓練預處理:

### resample_niigz.py

將input data統一成channel first以方便計算(若都已是(x,512,512)格式則跳過這步驟)

input:

(512,512,x)dicom的nii.gz檔

(512,512,x)mask的nii.gz檔

output:

(x,512,512)dicom的nii.gz檔

(x,512,512)mask的nii.gz檔

補充:注意旋轉方向，5,6行換成自己要的input path(dicom path,mask path)

    python resample_niigz.py

### 1_prepare_liver_and_maska.py

將dicom的nii.gz轉成三個不同的window level的npy檔

input:

dicom_for_liver_seg:(512,512,x)dicom的nii.gz檔

mask_for_liver_seg:(512,512,x)mask的nii.gz檔

output:

dicom的npy檔

補充:8,9行換成自己要的input path

    python 1_prepare_liver_and_maska.py

## 訓練:

### 2.5_liver_segmentation.py:

訓練權重，並輸出圖表(在權重檔內)

-e [int] epoch數

-p [str] pretrain weight 路徑(路徑後面都必須加'/')

-w [str] weight 名稱(路徑後面都必須加'/')

    python 2.5_liver_segmentation.py -e 100 -p weight_liver_25_726_200/ -w weight_liver_new/ 

## 檢視模型效能:

### 3.5_predict_liver.py:

拿test_data預測出mask並輸出nii.gz檔，和圖片對照(檢查預測結果用，若test data沒有mask請跳過這個步驟)

-w [str] 使用的權重路徑(路徑後面都必須加'/')

-d [str] 結果輸出路徑(路徑後面都必須加'/')

補充:132,133行換成需要test_data,test_mask的路徑

    python 3.5_predict_liver.py -w weight_liver_new/ -d predict_mask/

## 預測:

### 4.5_predict_mask_Toniigz.py:

拿test_data預測出mask並輸出nii.gz檔

input:

new_nii.gz:(x,512,512)dicom  nii.gz

output:

(x,512,512)mask nii.gz

-w [str] 使用的權重路徑(路徑後面都必須加'/')

-d [str] 結果輸出資料夾(路徑後面都必須加'/')

補充:78行換成input data的路徑(預設為new_nii.gz)

    python 4.5_predict_liver.py -w weight_liver_new/ -d predict_mask/

