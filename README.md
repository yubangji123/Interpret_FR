# Interpret_FR

This repository will give the training and evaluation code for the interpretable face recognition. For details, you can check  the associated paper [Towards Interpretable Face recognition](https://arxiv.org/abs/1805.00611).

## Pre-Requisites:
1. Tensorflow r1.0 or higher version
2. Python 2.7/3.6
3. Download CASIA-Webface training database [here](https://pan.baidu.com/s/19usJ4vjh__xVbmTv_GZUmQ) and  [here](https://pan.baidu.com/s/1Alm6gI-h5NaD5LEeUzPTZQ)
4. Download IJB-A testing database [here](https://pan.baidu.com/s/1Z_C0bD9MhBfGc9jw6K26ow)
5. Download auxiliary files for training  [here](https://pan.baidu.com/s/1_trzUM5rvphwVxYRKvc5UQ)
6. Download auxiliary files for evaluation  [here](https://pan.baidu.com/s/1R50FWhQNVXtvHzIy50Onow) and  [here](https://pan.baidu.com/s/13cswNI8fhn2SZIpolNYZPQ)

## Procedure to training the model:
1. Download the [model](https://pan.baidu.com/s/1tjyzS8Nb22MK45ymRYF2ag), unzip it and put the models in **/train/Interp_FR/CASIA_64_96_32_320_32_320/** directory.
2. Unzip the auxiliary files for training, and put all the **.mat** files in **/train/** directory.
2. For training data, put **CASIA_all_images_110_110.DAT** and **CASIA_recrop_fileList.DAT** in **DATA** folder.
3. Within the training folder, you can start to train the model through: `python main_pretrain_CASIA.py --dataset CASIA --batch_size 64 --is_train True --learning_rate 0.001 --image_size 96 --is_with_y True --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 320 --z_dim 20 --checkpoint_dir ./Interp_FR --gpu 0`.


## Procedure to evaluation and visualization:
Evaluate the face recognition performance:
1. Clone the Repository to preserve Directory Structure
2. For IJB-A, put **IJBA_recrop_images_96_96_test.DAT** in **DATA** folder.
3. Generate features(features.npy) corresponding to the images in the dataset folder by running: `python test_pre_train.py`.
4. After step 3, you will get a **IJBA_features_iter_0.txt** file, which contains all the face features.  To evaluate the results, you should put the auxiliary files **dataset.mat** and **IJBA_crop.mat** in **/test/eval/IJB-A/** directory.
5. By runing the **CNN_single_verify.m** and **CNN_single_search.m** script files, you will get the verification and identification results on IJB-A.

Visualize the average locations of peak response:
1. Clone the Repository to preserve Directory Structure
2. For IJB-A, put **IJBA_recrop_images_96_96_test.DAT** in **DATA** folder.
3. Generate features(features.npy) corresponding to the images in the dataset folder by running: `python test_pre_train.py`, you should comment the line `extract_features_for_eval()`. In this repository, we only frozen ours model, for base CNN and spatial only models, you can use the provided **freeze_my_model.py** to freeze the required models.
4. After step 3, you will get a **IJBA_feature_maps*.txt** file, which contains all the face feature maps.  To evaluate the results, you should put all three needed **.txt** files in **/test/visualization/** directory.
5. By runing the **average_location.m** script files, you will get 3 figures for the average locations of the models.


## Citation:

If you use our model or code in your research, please cite the paper:

```
@inproceedings{InterpretFR2018,
  title={Towards Interpretable Face recognition},
  author={Bangjie Yin and Luan Tran and Haoxiang Li and Xiaohui Shen and Xiaoming Liu},
  booktitle={arXiv preprint arXiv:1805.00611},
  year={2018}
}
```