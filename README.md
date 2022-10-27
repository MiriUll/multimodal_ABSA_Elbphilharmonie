#Multimodal aspect-based sentiment sentiment analysis on Elbphilharmonie posts

This repository contains part of the code for our paper "Structuring User-Generated Content on Social Media with Multimodal Aspect-Based Sentiment Analysis".

In the ```scraping/``` folder, the code for scraping the data form Flickr can be found as well as the dataset used for our study. Due to the size of the data, we do not put the images in this repository (except for the test dataset). The images can be downloaded with the ```scraping/download_images.py``` script.

To download and train the used models, refer to their original Github pages at (https://github.com/naver/deep-image-retrieval)[https://github.com/naver/deep-image-retrieval] and (https://github.com/cbaziotis/datastories-semeval2017-task4)[https://github.com/cbaziotis/datastories-semeval2017-task4], respectively.

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
│   │   test_dataset.csv               
│   └───retrieval_full_evaluation      Closest and farthest Elbphilharmonie images in full dataset
│   └───scraping                       scrapy code for scraping
│   └───test_data                      Images in the image test dataset
│   └───test_data_evaluation           Test dataset AP model features (r-mac feature files) and visualization of different methods
```