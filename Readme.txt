���{���X�i�Ω�q��´�f�z�v���w���w�̭��n��]������{�A����s�H�j�z���z�����ҡA�HTCGA-CRC-DX���V�m�ACPTAC-COAD�MPAIP�@���~����ƶ����ҡC

�@�~�t��: Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-72-generic x86_64)
�{���s�边:VScode version 1.91
�������A��:ASUS ESC4000 G4 

####################

�˥����U��]���P�L�ìP��í�w�ʪ��A��z��H�Uxlsx��:
MSI_CRC_DX_0307.xlsx
MSI_CPTAC.xlsx
MSI_PAIP.xlsx
Gene_CRC_DX_0307.xlsx
Gene_CPTAC_0307.xlsx

####################

1. �w��C�i�v������e�B�z�B�J�A�N�C�iWSI�����h�ipatches�A�åB�ھ�Pixel intensity, Entropy�MCanny�i��z��A����:

python tile_segmentation.py CRC.txt

CRC.txt��TCGA-CRC-DX�������˥��A�i���tile_segmentation.py�����v���ӷ��P�x�s���|�C

2. ���ۦbMancenko_normaiztion���A����:

python Normalize.py -ip inputpath -op outputpath -si sample image path -nt threads

sample image�ϥ�preprocessing/Macenko_normalization/normalization_template.jpg

3. �غcTumor detetction model

�ϥ�NCT-CRC-HE-100K�MCRC-VAL-HE-7K��ƶ��A�HResNet18�i��V�m�A����:

python tumor_detection0222.py

�[��ҫ�������{�O�A�èϥ�Inference.py�N�V�m�n���ҫ����Ψ�TCGA-CRC-DX�BCPTAC�MPAIP���C

python inference.py

#####################

4. �HResNet50�@��feature extractor�A�N�C�ipatch���Y��2048x1���S�x�V�q�AResNet50����j�q��´�f�z�v���i��Self-supervised learning��o���ҫ��v���A�N�V�q�s��npy�ɡC

python feature_extractor_ssl.py

5. �N�L�o�n��patches�A�ϥ�Hover-Net��o�C�ipatch�ӭM�����ӼơA�ñN�ӭM�ӼƦP�˾�z��.npy�ɡC

./run_tile.sh CRC.txt pannuke

python feature_extractor_count.py

#####################

6. �ҫ��V�m

����AttMIL.ipynb�BHoverAtt.ipynb�MHoverAtt_CMS.ipynb��biomarker�i��w���C
AttMIL���Ȩϥμv���i��w�����ҫ��AHoverAtt�MHoverAtt_CMS.ipynb�����[�J�ӭM�����Ӽƪ��ҫ��[�c�C�䤤�ݧ��S�x�V�q���ӷ���m�A�Mhyperparamters���ƭȡC


7. ��ı��

���z�Lresized.py��o�C�@�iWSI�Y�p�᪺png�v���C

python resized.py

�A�z�Lvisualization.py�A�����ϥΪ��ҫ��v���ɮ׸��|�Ppatch�x�s��m�A�ھڪ`�N�O���ƪ����C�A�N���n���ϰ��ı�ơC

python visualization.py

�bvisualization.py���A�i�H����SVM��o���P�ӭM������biomarker���������n�{�סC


















