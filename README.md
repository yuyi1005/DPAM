# DPAM
A pytorch implementation of "Dynamic Points Agglomeration for Hierarchical Point Sets Learning" (DPAM) (ICCV2019)

Only the Point Cloud Classification part is available in this version, and the Parameter Sharing Scheme in Section 3.4 and T-Net in Section 4.1 are not yet implemented. 

According to Table 4 and Table 5, the accuracy should be about 90% on ModelNet40. However, our implementation reaches only 87.3% classification accuracy. We hope to discuss with everyone interested in this project.

The paper can be found in: 
http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Dynamic_Points_Agglomeration_for_Hierarchical_Point_Sets_Learning_ICCV_2019_paper.html

## Setup
### Preparation
Prepare binary file \*.bc for training (e.g. H:\ModelNet40\train and H:\ModelNet40\val).
Or modify the dataset.py file to load your data.

### Training
python train.py --data_dir=H:\ModelNet40

## Performance
### Classification on ModelNet40

|Model|Accuracy|
|-|-|
|DPAM+111 (Paper)|90.9%|
|DPAM+841 (Paper)|91.9%|
|DPAM(vanilla)+841 (Paper)|91.4%|
|DPAM(vanilla)+111 (This implementation)|87.3%|
