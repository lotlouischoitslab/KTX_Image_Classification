# KTX Image Classification
## Contributors: 
- ### Louis Sungwoo Cho (조성우)

# Project Description
This project is about image classifcation of the high-speed trains in South Korea, analyzing and forecasting KTX (Korea Train eXpress) (한국고속철도) and SRT (Super Rapid Train) (수도권고속철도) passenger ridership and the utility rate. Random image datasets were given into the neural network model. The combined passenger ridership datasets used for analyzing and forecasting were acquired from KORAIL (한국철도공사) and SRT (수서고속철도주식회사). 

![title](images/ktx_one.png)
### KTX-1 the original French TGV model high-speed train approaching a station.
### 역으로 들어오는 프랑스에서 제작한 TGV 모델 KTX-1 고속열차.
- #### Image Source: https://en.wikipedia.org/wiki/Korea_Train_Express

![title](images/ktx_sancheon.png)
### KTX-Sancheon model developed by Hyundai Rotem traveling along the Gangneung Line.
### 강릉선을 고속으로 주행하는 현대로템에서 제작한 KTX-산천 고속열차.
- ##### Image Source: https://www.archyworldys.com/only-56-minutes-from-cheongnyangni-to-jecheon-the-faster-and-strongerbullet-train-comes/

![title](images/ktx_eum.png)
### KTX-EUM model developed by Hyundai Rotem passing Yangsu Bridge of the Jungang Line.
### 중앙선 양수철교 구간을 고속으로 통과하는 현대로템에서 제작한 KTX-이음 고속열차.
- #### Image Source: http://www.greenpostkorea.co.kr/news/articleView.html?idxno=69229

![title](images/srt_train.png)
### SRT train developed by Hyundai Rotem passing Pyeongtaek Jije Station. 
### 평택지제역을 통과하는 현대로템에서 제작한 SRT 고속열차.
- #### Image Source: https://www.srail.or.kr/cms/article/view.do?postNo=39&pageId=KR0502000000

# Motivation
South Korea first opened their high-speed rail network on April 1st, 2004 to make rail travel time more fast and convenient. When I first traveled to South Korea, I still remember when I took KTX with my family for the first time when we went to Busan. I was excited to ride the high-speed train because the U.S does not have bullet trains which can travel as fast as the KTX trains. After nearly 2 decades the first KTX line the Gyeongbu High-Speed Line (경부고속선) connecting Seoul to Busan opened, the high-speed rail network has expanded almost throughout the entire country including the Honam High-Speed Line (호남고속선) connecting Seoul to Gwangjusongjeong to Mokpo, Suseo High-Speed Line or Sudogwon High-Speed Line (수서고속선/수도권고속선) connecting the south side of Seoul Suseo to Busan to Gwangju to Mokpo, Gyeongjeon Line (경전선) connecting Seoul to Masan to Jinju, Jeolla Line (전라선) connecting Seoul to Yeosu-EXPO, Donghae Line (동해선) connecting Seoul to Pohang, Gangneung Line (강릉선) also known as the 2018 Pyeongchang Olympics Line connecting Seoul to Gangneung, Yeongdong Line (영동선) connecting Seoul to Donghae, Jungang Line (중앙선) connecting Seoul to Andong (sections to Uiseong, Yeongcheon, Singyeongju, Taehwagang, Busan-Bujeon to be opened in December 2023), and the Jungbu-Naeryuk Line (중부내륙선) connecting Bubal to Chungju. As seen above, due to the continuing expansion of the South Korean high-speed train network, Hyundai ROTEM has designed many different types of variants to serve in various lines depending on their operational speed respectively. Due to each locomotive having unique features, I decided to create a deep learning model that can classify the 4 types of trains: KTX-1, KTX-EUM, KTX-Sancheon, and SRT. 

![title](images/ktx.png)
#### From left to right KTX-1, KTX-Sancheon, SRT, KTX-EUM (왼쪽부터 KTX-1, KTX-산천, SRT, KTX-이음)
- #### Image Source: https://www.youtube.com/watch?v=pSFV4Nh2hJo

![title](images/ktx_network.png)
### Map of the entire high-speed rail network in South Korea (대한민국 고속철도망)
- #### Image Source: https://www.incheontoday.com/news/articleView.html?idxno=205643

# High-Speed Train Information
The following section includes the information for each train with their vehicle specifications. KTX stands for Korea Train eXpress and SRT stands for Super Rapid Train.
# KTX-1
![title](images/ktx_one_info.png)

### Manufacturer: Alstom & Hyundai ROTEM
### Family: TGV
### Entered Service: April 2004
### Operator: KORAIL
### Lines Served: Gyeongbu HSR, Honam HSR, Gyeongjeon, Jeolla, Donghae
### Maximum Operating Speed: 305 km/h
### Maximum Design Speed: 330 km/h
### Electricity: 25 kV AC 60 Hz Catenary
### Current Collector: Pantograph
### Safety System: ATS, ATP, TVM-430
### Track Gauge: 1435 mm Standard Gauge

# KTX-Sancheon (KTX-산천)
![title](images/ktx_sancheon_info.png)

### Manufacturer: Hyundai ROTEM
### Family: KTX
### Entered Service: March 2010
### Operator: KORAIL
### Lines Served: Gyeongbu HSR, Honam HSR, Gyeongjeon, Jeolla, Donghae
### Maximum Operating Speed: 305 km/h
### Maximum Design Speed: 330 km/h
### Electricity: 25 kV AC 60 Hz Catenary
### Current Collector: Pantograph
### Safety System: ATS, ATP, TVM-430
### Track Gauge: 1435 mm Standard Gauge

# SRT
![title](images/srt_info.png)

### Manufacturer: Hyundai ROTEM
### Family: KTX
### Entered Service: December 2016
### Operator: SR Corporation
### Lines Served: Suseo/Sudogwon HSR, Gyeongbu HSR, Honam HSR
### Maximum Operating Speed: 305 km/h
### Maximum Design Speed: 330 km/h
### Electricity: 25 kV AC 60 Hz
### Current Collector: Pantograph
### Safety System: ATS, ATP, TVM-430
### Track Gauge: 1435 mm Standard Gauge

# KTX-EUM (KTX-이음)
![title](images/ktx_eum_info.png)

### Manufacturer: Hyundai ROTEM
### Family: KTX
### Entered Service: January 2021
### Operator: KORAIL
### Lines Served: Gangneung, Yeongdong, Jungang, Jungbunaeryuk, Seohae (Planned)
### Maximum Operating Speed: 260 km/h
### Maximum Design Speed: 286 km/h
### Electricity: 25 kV AC 60 Hz Catenary
### Current Collector: Pantograph
### Safety System: ATS, ATP, TVM-430
### Track Gauge: 1435 mm Standard Gauge

## Information Sources:
- ##### https://en.wikipedia.org/wiki/KTX-I
- ##### https://en.wikipedia.org/wiki/KTX-Sancheon
- ##### https://en.wikipedia.org/wiki/Super_Rapid_Train
- ##### https://en.wikipedia.org/wiki/KTX-Eum

# Image Preparation
Random South Korean high-speed train image datasets were used to train the neural network model for image classification. 40 files were then split into 4 categories with each category having 10 images of the same class. 

![title](images/random_ktx_one.png)
### Figure 1. above shows the 10 random KTX-1 images from the given image dataset.

![title](images/random_ktx_eum.png)
### Figure 2. above shows the 10 random KTX-EUM images from the given image dataset.

![title](images/random_ktx_sancheon.png)
### Figure 3. above shows the 10 random KTX-Sancheon images from the given image dataset.

![title](images/random_srt.png)
### Figure 4. above shows the 10 random SRT images from the given image dataset.

Once all the random image datasets were printed out, the entire image dataset was split into training and testing sets. 80% of the total image datasets were used for training and the remaining 20% of the total image datasets were used for testing. The epochs number was set to 20 so the training model was run for 20 times. Then all the data were shuffled before the neural network model was created. 

# Convolutional Neural Network (CNN) Model
Convolutional Neural Network (CNN) model was used to classify the high-speed train images. One of the biggest advantage of using CNN models is that the neural network is able to detect the important features into several distinct classes from the given image datasets without any human supervision and also being much more accurate and computationally efficient than Artifcial Neural Networks (ANN). Hence, this deep learning model was chosen to train all the high-speed trains image datasets for this project. 

![title](images/cnn_process.png)
#### Figure 5. above shows how the cnn model processes the image dataset with series of convolution and pooling before flattening out the image to predict the output.

The model used for this project performs multiclass classification so the output is set to be softmax. But why is convolution so crucial in image classification? Convolution is a set of mathematical operations performed by the computer to merge two pieces of critical information from the image. A feature map for the images is produced using a 'convolution filter'. 

![title](images/cnn_filter.png)
#### Figure 6. above shows how the convolution filter produces the feature map.

The convolution operation is then performed by splitting the 3 by 3 matrix into merged 3 by 3 matrix by doing an element-wise matrix multiplication and summing the total values. 

![title](images/cnn_matrix.gif)
#### Figure 7. above shows the matrix operation of the convolution filter.

![title](images/cnn_visual.gif)
#### Figure 8. above shows the visualization of the  convolution input of the image.

Once all the convolution has been performed on the image datasets, pooling is then used to reduce the dimensions, a crucial step to enable reducing the number of parameters shortening the training time and preventing overfitting. Maximum pooling was used for this model which only uses the maximum value from the pooling window. 

![title](images/cnn_pooling2d.png)
#### Figure 9. above shows the pooling of the processed image in a 2 by 2 window.

![title](images/cnn_pooling3d.png)
#### Figure 10. above shows the pooling of the processed image in a 3 by 3 window.

Finally after adding all the convolution and pooling layers, the entire 3D tensor is flatten out to be a 1D vector into a fully connected layer to produce the output. 
![title](images/cnn_imp.png)
#### Figure 11. above shows the visual implementation of the CNN model. 

##### Original Source for the CNN Explanation: https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2#:~:text=The%20main%20advantage%20of%20CNN,CNN%20is%20also%20computationally%20efficient.

# Results
Once the CNN model was built for image classification training with a given number of training steps also known as epochs set to 20, the accuracy score graph and the loss score graph with respect to each epoch step were plotted. 

![title](images/accuracy.png)
#### Figure 12. above shows the accuracy score of the CNN model with respect to the number of steps. 

![title](images/loss.png)
#### Figure 13. above shows the loss score of the CNN model with respect to the number of steps. 

According to the plots above, the train accuracy is very close to the testing accuracy as the number of epochs gradually increases. Overall, the model has produced a relatively high training accuracy. The number of losses meaning the error between the actual image and the predicted image decreases as more number of epochs are given into the model. This means that the chance of predicting a given image dataset accurately is very high. 

# Prediction
Once all the image datasets have been processed and the accuracy and loss score have been analyzed, a few set of images were given into the model to determine whether the model is accurate enough predicting the train type of a given image. 8 testing datasets were given into the model and the predictor plots the actual image and the predicted image with a confidence score respectively. 

![title](images/predicted_output.png)
#### Figure 14. above shows the predicted output of each image data given into our model with the train type and the confidence score for each image.

From the image above, it is clearly evident that the predictor estimates the train class very accurately. The confidence is also very high for each results meaning that the results have turned out very well. Overall, the model performed very well with all the high-speed train image datasets.

# Passenger Ridership Analysis
This was the second part of the project. Datasets that include total passengers of each high-speed line and utility rate acquired from KTX and SRT were used to analyze and forecast total passengers ridership and utility rate. The number of passengers is represented in thousands.
- #### Dataset Source: https://www.index.go.kr/potal/main/EachDtlPageDetail.do?idx_cd=1252

### Utility Rate (UR) Formula:
- ##### passnum = actual number of passengers in thousands who boarded the train
- ##### availseats = number of seats available

$$ UR\ =\ \frac{passnum}{availseats} \*\ 100 \\% \ $$

This is the formula defined by both KTX and SRT companies and the utility rate datasets were already calculated and given in the raw data so no extra work had to be done to compute the UR value for each high-speed train line. 

# Data Visualization and Forecasting

Because there are too many variables to plot in one graph from the raw data, two new dataframes have been created to analyze and forecast the total number of passengers on each line and utility rate. 

![title](data_explore/passen_analysis.png)
#### Figure 15. above shows the number of passengers for each line and the total passengers for each month.

![title](data_explore/utility_analysis.png)
#### Figure 16. above shows the utility rate for each line and the total utility rate of each train for each month.

![title](data_explore/passen_est.png)
#### Figure 17. above shows the total passengers for each line and for each train for each month.

![title](data_explore/utility_est.png)
#### Figure 18. above shows the utility rate for each line and the utility rate for each train for each month.

Once all the estimators have been added to approximate the trend for each graph, time-series forecasting was performed to predict future data using the trained datasets. The results were then plotted below.

![title](data_explore/passen_pred.png)
#### Figure 19. above shows the predicted number of passengers of each line until 2030.

![title](data_explore/utility_pred.png)
#### Figure 20. above shows the predicted utility rate of each line until 2030.

# Conclusive Remarks
Overall, the convolutional neural network model used for image classification has performed very well classifying the 4 different types of Korean high-speed trains: KTX-1, KTX-EUM, KTX-Sancheon, and SRT with high confidence score. It is very evident that convolutional neural networks have strong computational power while producing accurate results when classifying the images. More epochs and more layers overall improved the accuracy for this model. For further improvements in image classification, adding more complex samples to the model will be considered. 

However, the time-series forecasting model did go a little off with predicting the future passenger numbers on the Gyeongbu High-Speed Line. This is probably due to the sudden drop during the COVID-19 pandemic from 2020 to 2022 onwards and the fact the high-speed rail companies do not have all the datasets collected in 2022. To further improve predicting the passenger ridership trends, more passenger datasets will be collected in the following years.

![title](louis_seoul_station/signs.jpg)
## 대전, 동대구, 부산, 포항, 진주, 마산, 강릉, 동해 방면
### To Daejeon, Dongdaegu, Busan, Pohang, Jinju, Masan, Gangneung, Donghae

## 익산, 광주송정, 목포, 여수EXPO 방면
## To Iksan, Gwangjusongjeong, Mokpo, Yeosu-EXPO

![title](louis_seoul_station/ktx_eum.png)
# Have a fun trip everyone!
# 즐거운 여행이 되세요!
