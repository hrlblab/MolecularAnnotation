# Democratizing Pathological Image Segmentation with Lay Annotators via Molecular-empowered Learning

### [[Project Page]](https://https://github.com/ddrrnn123/Omni-Seg/)   [[MICCAI2023 Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_48)


This is the official implementation of Democratizing Pathological Image Segmentation with Lay Annotators via Molecular-empowered Learning. 

![Overview](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview1.png)<br />
![Docker](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview2.png)<br />

**MICCAI2023 Paper** <br />
> [Democratizing Pathological Image Segmentation with Lay Annotators via Molecular-empowered Learning](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_48) <br />
> Ruining Deng, Yanwei Li,Peize Li,Jiacheng Wang, Lucas W. Remedios, Saydolimkhon Agzamkhodjaev, Zuhayr Asad, Quan Liu, Can Cui, Yaohong Wang, Yihan Wang, Yucheng Tang, Haichun Yang, Yuankai Huo.<br />
> *International Conference on Medical Image Computing and Computer-Assisted Intervention 2023* <br />

## Abstract
Multi-class cell segmentation in high-resolution Giga-pixel whole slide images (WSI) is critical for various clinical applications. Training such an AI model typically requires labor-intensive pixel-wise manual annotation from experienced domain experts (e.g., pathologists). Moreover, such annotation is error-prone when differentiating fine-grained cell types (e.g., podocyte and mesangial cells) via the naked human eye. In this study, we assess the feasibility of democratizing pathological AI deployment by only using lay annotators (annotators without medical domain knowledge). <br /> 


The contribution of this paper is threefold: <br />
(1) We proposed a molecular-empowered learning scheme for multi-class cell segmentation using partial labels from lay annotators; <br />
(2) The proposed method integrated Giga-pixel level molecular-morphology cross-modality registration, molecular-informed annotation, and molecular-oriented segmentation model, so as to achieve significantly superior performance via 3 lay annotators as compared with 2 experienced pathologists; <br />
(3) A deep corrective learning (learning with imperfect labels) method is proposed to further improve the segmentation performance using partially annotated noisy data. <br />

From the experimental results, our learning method achieved F1 = 0.8496 using molecular-informed annotations from lay annotators, which is better than conventional morphology-based annotations (F1 = 0.7015) from experienced pathologists. Our method democratizes the development of a pathological segmentation deep model to the lay annotator level, which consequently scales up the learning process similar to a non-medical computer vision task. <br />

## Multi-modality Registration
Run the Python scripts as the following orders to achieve multi-modality registration (originally from 3D multi-stain pathological image registration pipeline [Map3D](https://github.com/hrlblab/Map3D)): <br />
1. Get PNGs from WSIs in 5X. <br />
[Step1_WSI_to_png_5X.py](Multi-modalityRegistration/Step1_WSI_to_png_5X.py) <br />
2. Global alignment by SuperGlue. <br />
[Step2_superglue.py](Multi-modalityRegistration/Step2_superglue.py) <br />
3. Apply global alignment to check the slide-wise registration performance. <br />
[Step3_ApplySGToMiddle.py](Multi-modalityRegistration/Step3_ApplySGToMiddle.py) <br />
4. Prepare initial affine matrix for ANTs registration. <br />
[Step4_matrix_npytomat_5X.py](Multi-modalityRegistration/Step4_matrix_npytomat_5X.py) <br />
5. Global alignment by ANTs. <br />
[Step5_SuperGlue+ANTs.py](Multi-modalityRegistration/Step5_SuperGlue+ANTs.py) <br />
6. Apply second-step global alignment to check the slide-wise registration performance. <br />
[Step6_BigRecon_moveAllslicesToMiddle_IHCtoPAS.py](Multi-modalityRegistration/Step6_BigRecon_moveAllslicesToMiddle_IHCtoPAS.py) <br />
7. Crop regions by using an affine matrix from either SuperGlue or Ants. <br />
[Step7.5_matrix_20Xmat_affine_4points_WSI_SGOnly.py](Multi-modalityRegistration/Step7.5_matrix_20Xmat_affine_4points_WSI_SGOnly.py) <br />
[Step7_matrix_20Xmat_affine_4points_WSI_SG+ANTs.py](Multi-modalityRegistration/Step7_matrix_20Xmat_affine_4points_WSI_SG+ANTs.py)  <br />
8. pixel-level registration by AIRLab. <br />
[Step8_airlab_affine_registration_2d_PAS_2048.py](Multi-modalityRegistration/Step8_airlab_affine_registration_2d_PAS_2048.py) <br />

## Molecular-oriented corrective learning for partial label segmentation
The segmentation pipeline is originally from a single dynamic network for partially labeled dataset [Omni-Seg](https://github.com/ddrrnn123/Omni-Seg).
1. Use [train_2D_patch_omni-seg_selfloss.py](CorrectiveLearning/train_2D_patch_omni-seg_selfloss.py) to train the model.
2. Use [Testing_2D_patch_omni-seg_ns.py](CorrectiveLearning/Testing_2D_patch_omni-seg_ns.py) to test the model.

## Citation
```
@inproceedings{deng2023democratizing,
  title={Democratizing Pathological Image Segmentation with Lay Annotators via Molecular-Empowered Learning},
  author={Deng, Ruining and Li, Yanwei and Li, Peize and Wang, Jiacheng and Remedios, Lucas W and Agzamkhodjaev, Saydolimkhon and Asad, Zuhayr and Liu, Quan and Cui, Can and Wang, Yaohong and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={497--507},
  year={2023},
  organization={Springer}
}

```


