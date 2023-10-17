# Democratizing Pathological Image Segmentation with Lay Annotators via Molecular-empowered Learning

[[Project Page]](https://https://github.com/ddrrnn123/Omni-Seg/)   [[MICCAI2023 Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_48)


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
(3) A deep corrective learning (learning with imperfect label) method is proposed to further improve the segmentation performance using partially annotated noisy data. <br />

From the experimental results, our learning method achieved F1 = 0.8496 using molecular-informed annotations from lay annotators, which is better than conventional morphology-based annotations (F1 = 0.7015) from experienced pathologists. Our method democratizes the development of a pathological segmentation deep model to the lay annotator level, which consequently scales up the learning process similar to a non-medical computer vision task. <br />



