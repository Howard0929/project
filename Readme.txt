本程式碼可用於從組織病理影像預測患者重要基因型的表現，本研究以大腸直腸癌為例，以TCGA-CRC-DX做訓練，CPTAC-COAD和PAIP作為外部資料集驗證。

作業系統: Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-72-generic x86_64)
程式編輯器:VScode version 1.91
網路伺服器:ASUS ESC4000 G4 

####################

樣本的各基因型與微衛星不穩定性狀態整理於以下xlsx檔:
MSI_CRC_DX_0307.xlsx
MSI_CPTAC.xlsx
MSI_PAIP.xlsx
Gene_CRC_DX_0307.xlsx
Gene_CPTAC_0307.xlsx

####################

1. 針對每張影像執行前處理步驟，將每張WSI切成多張patches，並且根據Pixel intensity, Entropy和Canny進行篩選，執行:

python tile_segmentation.py CRC.txt

CRC.txt為TCGA-CRC-DX全部的樣本，可更改tile_segmentation.py中的影像來源與儲存路徑。

2. 接著在Mancenko_normaiztion中，執行:

python Normalize.py -ip inputpath -op outputpath -si sample image path -nt threads

sample image使用preprocessing/Macenko_normalization/normalization_template.jpg

3. 建構Tumor detetction model

使用NCT-CRC-HE-100K和CRC-VAL-HE-7K資料集，以ResNet18進行訓練，執行:

python tumor_detection0222.py

觀察模型份類表現力，並使用Inference.py將訓練好的模型應用到TCGA-CRC-DX、CPTAC和PAIP中。

python inference.py

#####################

4. 以ResNet50作為feature extractor，將每張patch壓縮成2048x1的特徵向量，ResNet50為對大量組織病理影像進行Self-supervised learning獲得的模型權重，將向量存成npy檔。

python feature_extractor_ssl.py

5. 將過濾好的patches，使用Hover-Net獲得每張patch細胞種類個數，並將細胞個數同樣整理成.npy檔。

./run_tile.sh CRC.txt pannuke

python feature_extractor_count.py

#####################

6. 模型訓練

執行AttMIL.ipynb、HoverAtt.ipynb和HoverAtt_CMS.ipynb對biomarker進行預測。
AttMIL為僅使用影像進行預測的模型，HoverAtt和HoverAtt_CMS.ipynb為有加入細胞種類個數的模型架構。其中需更改特徵向量的來源位置，和hyperparamters的數值。


7. 視覺化

先透過resized.py獲得每一張WSI縮小後的png影像。

python resized.py

再透過visualization.py，更改欲使用的模型權重檔案路徑與patch儲存位置，根據注意力分數的高低，將重要的區域視覺化。

python visualization.py

在visualization.py中，可以執行SVM獲得不同細胞種類對biomarker分類的重要程度。


















