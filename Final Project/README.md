# Medical Image Classification with ResNets, DenseNets, EfficientNet and ViT
Medical image classification plays an increasingly important role in healthcare, especially in diagnosing, treatment planning, and disease monitoring. However, the lack of large publicly available datasets with annotations means it is still very difficult, if not impossible, to achieve clinically relevant computer-aided detection and diagnosis (CAD). In recent years, deep learning models have been shown to be very effective at image classification, and they are increasingly being used to improve medical tasks. Thus, this project aims to explore the use of different convolutional neural network (CNN) architectures for medical image classification. Specifically, we will examine the performance of 6 different CNN models (ResNet18, ResNet152, DenseNet121, DenseNet161, EfficientNet_B0, and ViT) on datasets of blood cell images and chest X-ray images.
### Datasets
We will use two datasets for our experiments:
1) [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells): This dataset contains 12,500 augmented images of blood cells with 4 different cell types, namely Eosinophil, Lymphocyte, Monocyte, and Neutrophil. 
We use this dataset for the multi-class classification problem.
2) [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample?select=sample_labels.csv): This dataset contains 5,606 chest X-rays from random patients with 15 classes (14 diseases, and one for "No findings").
We use this dataset for the multi-label classification problem.
### Loss function
Multi-class classification refers to the categorization of instances into precisely one class from a set of multiple classes. So, the commonly used loss function is cross-entropy loss. \
Multi-label classification involves instances that can belong to multiple classes simultaneously. Binary cross-entropy loss is commonly employed in this scenario.
### Models
1)	[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf), or Residual Network, is a deep convolutional neural network architecture that was introduced in 2015 by He et al. ResNets work by using residual connections to skip over layers in the network. This allows the network to learn more complex features without becoming too deep and overfitting to the training data.
2)	[DenseNet](https://arxiv.org/pdf/1608.06993v5.pdf), or Densely Connected Network, is another deep convolutional neural network architecture that was introduced in 2016 by Huang et al. DenseNets are similar to ResNets, but they use dense connections to connect all of the layers in the network. This allows the network to learn more global features and improve the accuracy of the model.
3)	[EfficientNet](https://arxiv.org/pdf/1905.11946v5.pdf) is a family of convolutional neural network architectures that were introduced in 2019 by Tan et al. EfficientNets are designed to be efficient in terms of both accuracy and computational resources. They achieve this by using a combination of techniques, including compound scaling, squeeze-and-excitation blocks, and autoML.
4)	[ViT](https://arxiv.org/pdf/2010.11929.pdf), or Vision Transformer, is a deep learning model that was introduced in 2020 by Dosovitskiy et al. Vision Transformers are based on the transformer architecture, which was originally developed for natural language processing (NLP).
### Evaluation
![image](https://github.com/JuliaKudryavtseva/dl2023_project/blob/main/Image/class_loss.jpg) 
![image](https://github.com/JuliaKudryavtseva/dl2023_project/blob/main/Image/label_loss.jpg) 
<p align="center">
  <img src="https://github.com/JuliaKudryavtseva/dl2023_project/blob/main/Image/class_acc.jpg" width="47%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/JuliaKudryavtseva/dl2023_project/blob/main/Image/label_acc.jpg" width="47%">
</p>

### Metrics
The results of our experiments are shown in two tables below.
1)	For multi-class classification problem:

| Model	| Accuracy | Precision (macro) |	Precision (micro) |	Recall (macro)	| Recall (micro) |	F1 (macro)	| F1 (micro)|
| :----------- | :----------- | :----------- |:----------- |:----------- |:----------- |:----------- | :-----------: |
|	resnet152	| 0.914757 |	0.913539	| 0.914757 | 0.914980 | 0.914757 | 0.913517 | 0.914757 |
| resnet18	| 0.932449 | 0.936440	| 0.932449 | 0.932577 |	0.932449 | 0.932841 |	0.932449 |
| densenet121	| 0.872939 | 0.887754 |	0.872939 | 0.872932 |	0.872939 | 0.874157 |	0.872939 |
|	densenet161	| 0.916365	| 0.917669 | 0.916365	| 0.916586	| 0.916365 | 0.915667 | 0.916365 |
| effnet	| 0.931242 |	0.933620 |	0.931242	| 0.931398 | 0.931242	| 0.931379 | 0.931242 |
| vit |	0.894250	| 0.912612 | 0.894250	| 0.894179	| 0.894250 | 0.895673 |	0.894250 |
* ResNet18 and EfficientNet_B0 are two best models with 0.93 accuracy and F1 score.
* DenseNet121 and DenseNet161 are unstable, accuracy and F1 are 0.87 and 0.91 correspondingly for both nets. 
* ViT showed SOTA results (accuracy: 0.89, F1: 0.89) with stable learning. 
2)	For multi-label classification problem:

| Model |	Exact match ratio	| Hamming loss |	Recall	| Precision |	F1 |
| :----------- |:----------- |:----------- |:----------- |:----------- | :-----------: |
| ResNet152	| 0.501338	| 0.501338	| 0.502230	| 0.501933 |	0.502052 |
|	ResNet18 |	0.473684 |	0.473684 |	0.476360 |	0.474874 | 0.475320 |
|	densenet121 |	0.513827 |	0.513827 | 0.513827 |	0.513827 | 0.513827 |
|	densenet161 |	0.520963 |	0.520963 | 0.520963 |	0.520963 | 0.520963 |
|	effnet | 0.520963 |	0.520963 | 0.520963 |	0.520963 | 0.520963 |
|	vit	| 0.520963 | 0.520963 | 0.520963 | 0.520963	| 0.520963 |

* DenseNets and EfficientNet_B0 showed satisfied results with 0.52 accuracy and F1 score
* DenseNets and ResNets were unstable, while EfficientNet_B0 showed stable learning.
* ViT with metrics, as other algorithms and showed stable learning.
### Conclusion
Overall, EfficientNet is the most efficient model for both multi-class and multi-label classification problems. It performs well in all metrics and shows stable learning. ViT performs as other algorithms in terms of metrics. However, there are significantly less params to train. That makes this architecture to be alternative for ConvNets in the future.
