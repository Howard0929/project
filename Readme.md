###### Molecular classification
This code can be used to predict the expression of important genotypes in patients from histopathological images. This study takes colorectal cancer as an example, using TCGA-CRC-DX for training and CPTAC-COAD and PAIP as external datasets for validation.

Operating System: Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-72-generic x86_64)
Code Editor: Python 3.10.9

### Dataset ###
The genotypes and microsatellite instability status of the samples are organized in the following xlsx files:

MSI_CRC_DX_0307.xlsx
MSI_CPTAC.xlsx
MSI_PAIP.xlsx
Gene_CRC_DX_0307.xlsx
Gene_CPTAC_0307.xlsx

### Methods ###

1. Perform preprocessing steps for each image, cutting each WSI into multiple patches, and filtering based on Pixel intensity, Entropy, and Canny.

        python tile_segmentation.py CRC.txt

   CRC.txt includes all samples of TCGA-CRC-DX. The image source and storage path in tile_segmentation.py can be modified.

3. Mancenko_normaiztion

        python Normalize.py -ip inputpath -op outputpath -si sample image path -nt threads

   Use preprocessing/Macenko_normalization/normalization_template.jpg as the sample image.

3. Tumor detetction model

   Train with the NCT-CRC-HE-100K and CRC-VAL-HE-7K datasets using ResNet18.

        python tumor_detection0222.py

   The model's classification performance can be observed by and apply the trained model to TCGA-CRC-DX, CPTAC, and PAIP using Inference.py.
    
        python inference.py

5. Use ResNet50 as the feature extractor, compressing each patch into a 2048x1 feature vector. ResNet50 weights are obtained through self-supervised learning on a large number of histopathological images. Save the vectors as npy files.

6. For filtered patches, use Hover-Net to obtain the cell count for each patch and organize the cell counts into .npy files.

        ./run_tile.sh CRC.txt pannuke

        python feature_extractor_count.py

7. Model training

     Execute AttMIL.ipynb, HoverAtt.ipynb, and HoverAtt_CMS.ipynb for biomarker prediction. AttMIL is a model that only uses images for prediction, while HoverAtt and HoverAtt_CMS.ipynb are models that    include cell count. Modify the source location of the feature vectors and the values of the hyperparameters as needed.

7. Visualization

   First, use resized.py to get the downscaled png image of each WSI.

        python resized.py

   Then, use visualization.py to change the path of the model weight file and patch storage location. Visualize important areas based on attention scores.

        python visualization.py

   In visualization.py, you can execute SVM to obtain the importance of different cell types in biomarker classification.

















