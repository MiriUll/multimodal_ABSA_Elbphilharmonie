# Multimodal aspect-based sentiment sentiment analysis on Elbphilharmonie posts

![pipeline](https://github.com/MiriUll/multimodal_ABSA_Elbphilharmonie/blob/main/data_pipeline_paper.png)

This repository contains part of the code for our paper "Retrieving Users' Opinions on Social Media with Multimodal Aspect-Based Sentiment Analysis" published at ICSC.

In the ```scraping/``` folder, the code for scraping the data form Flickr can be found as well as the dataset used for our study. Due to the size of the data, we do not put the images in this repository (except for the test dataset). The images can be downloaded with the ```scraping/download_images.py``` script.

To download and train the used models, refer to their original Github pages at https://github.com/naver/deep-image-retrieval and https://github.com/cbaziotis/datastories-semeval2017-task4, respectively.

## Repository structure
```
multimodal ABSA
│   README.md
│   remove_duplicates.ipynb            Notebook to summarize gallary posts
│   sentiment_analysis.ipynb           Notebook to try different sentiment classification approaches
│   sentiment_training.py              Train the models on the modified SemEval data
│   test_dataset_images.ipynb          Notebook to compare different feature extraction methods on the image test dataset
│   test_dataset_sentiment.ipynb       Notebook to evaluate the sentiment models on sentiment test dataset
│   vectorizer.py                      Script to vectorize test data
│   
└───scraping                           Code for scraping data and scraping results
│   │   download_images.py             Code to download the images based on scraped information
│   │   flickr_data_clean_2016.csv     Cleaned Flickr data for 2016
│   │   flickr_data_clean_2017.csv     Cleaned Flickr data for 2017
│   │   flickr_data_clean_2018.csv     Cleaned Flickr data for 2018
│   │   flickr_data_clean_2019.csv     Cleaned Flickr data for 2019
│   │   flickr_elphi_images.csv        Full dataset with Elbphilharmonie images (according to image retreival part)
│   │   flickr_elphi_images_translated.csv     Translated version of posts
│   │   main.py                        Scraping main file
│   │   scrapy.cfg                     scrapy configuration
│   │   test_dataset.csv               Posts belongig to images in test dataset, labeled with the landmark they show
│   │   test_dataset_sentiment.csv     Message-level sentiment test dataset
│   │   test_dataset_sentiment_aspect.csv       Aspect-based sentiment test dataset        
│   └───retrieval_full_evaluation      Closest and farthest Elbphilharmonie images in full dataset
│   └───scraping                       scrapy code for scraping
│   └───test_data                      Images in the image test dataset
│   └───test_data_evaluation           Test dataset AP model features (r-mac feature files) and visualization of different methods
```

## Recreate paper results
This project was build for Python 3.9. Use the command ```pip install -r requirements.txt``` to install the necessary packages. To load the sentiment prediction models, Tensorflow 2.6 is required.
### Feature vector visualization
We publish the images in the test dataset as well as the features (```scraping/test_data_evaluation/r-mac-features.npy```) with the best performing model. To visualize their clustering behavior in 2D, run this command:
```
python pca_vis_img_features.py
```
To use this script on custom data, the paths to the extracted features file and the dataset file have to be updated (lines 7 and 8). In addition, if you use other landmark classes, update the color mapping and class list (lines 9ff).

### Image retrieval
To use the image features to retrieve the closes images, run the following command:
```
python retrieve_closest_images.py
```
Again, if you want to adapt the script to your data, replace the paths in the beginning of the file with paths to your data.

### Sentiment analysis
The pretrained models are stored in the folder ```sentiment_models```. You need to download them using git lfs.
Then you can run the prediction on our sentiment test data with this command:
```
python predict_sentiment.py
```
If you want to run the prediction on custom data, the paths to the test dataset files need to be updated in the beginning of the file.

### Review experiments
To recreate the comparison of different image retrieval techniques and sentiment analysis methods, refer to the Jupyter notebooks [test_dataset_images.ipynb](https://github.com/MiriUll/multimodal_ABSA_Elbphilharmonie/blob/main/test_dataset_images.ipynb) and [test_dataset_sentiment.ipynb](https://github.com/MiriUll/multimodal_ABSA_Elbphilharmonie/blob/main/test_dataset_sentiment.ipynb) respectively.
The full target dataset as well as the two testsets are provided in the folder ```scraping/```. Due to the size of the data, we only publish the images in the test dataset. The images for the full dataset need to be downloaded with the provided script.