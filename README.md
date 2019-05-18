# Interpret_FR

This repository only gives the testing code and frozen pretrained model. For details, you can check  the associated paper [Towards Interpretable Face recognition](https://arxiv.org/abs/1805.00611).

## Pre-Requisites:
1. Tensorflow r1.0 or higher version
2. Python 2.7/3.6
3. Download IJB-A database [here]()
4. Download auxiliary files  [here]() and  [here]()

## Procedure to Reproduce the Results:
1. Clone the Repository to preserve Directory Structure
2. For IJB-A, put **IJBA_recrop_images_96_96_test.DAT** in **DATA** folder.
3. Generate features(features.npy) corresponding to the images in the dataset folder by running: `python test_pre_train.py`
4. After step 3, you will get a **.txt** file, which contains all the face features.  To evaluate the results, you should put the auxiliary files **dataset.mat** and **IJBA_crop.mat** in **/eval/IJB-A/** directory.
5. By runing the **CNN_single_verify.m** and **CNN_single_search.m** script files, you will get the verification and identification results on IJB-A.

## Citation:

If you use our model or code in your research, please cite the paper:

```
@article{Mathur2017,
  title={Camera2Caption: A Real-time Image Caption Generator},
  author={Pranay Mathur and Aman Gill and Aayush Yadav and Anurag Mishra and Nand Kumar Bansode},
  journal={IEEE Conference Publication},
  year={2017}
}
```
