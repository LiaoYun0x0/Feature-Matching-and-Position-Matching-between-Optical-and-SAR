# Feature Matching and Position Matching between Optical and SAR with Local Deep Feature Descriptor



Using Deeplearning to locate the Synthetic Aperture Radar(SAR) images to the counterpart of Optical images.


Model backbone: CSP + Dense Block CNN
Loss: Arc loss and l2 loss



Demo:

run:
python location_demo.py

image pairs in 'test_image' will be loaded into model and predict the location of SAR images in optical images.
The matching results will save as images.



# Chinese
利用CSP Block 和 Dense Block 搭建的CNN网络实现对光学图像和合成孔径雷达图像的位置匹配。

通过使用模型提取的图像关键点特征描述子进行异源图像匹配

示例：
python location_demo.py
读入'test_image'中待匹配的图像进行匹配，结果以图片形式保存

![image](https://github.com/LiaoYun0x0/Feature-Matching-and-Position-Matching-between-Optical-and-SAR/blob/main/1_1.00_1_kset.png)
![image](https://github.com/LiaoYun0x0/Feature-Matching-and-Position-Matching-between-Optical-and-SAR/blob/main/1_1.00_1_kset_circle.png)
