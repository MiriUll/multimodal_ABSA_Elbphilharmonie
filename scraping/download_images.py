import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
from urllib.error import HTTPError
import argparse

parser = argparse.ArgumentParser(description='Download images from Flickr')
parser.add_argument('--data_path', '-d', type=str, help='path to the csv file with the data',
                    default='flickr_data_01_2016.csv')
parser.add_argument('--image_folder', '-i', type=str, help='the folder in which the images are stored', default='img')
parser.add_argument('--start_index', '-s', type=int, help='A custom start index if som of the data has already been '
                                                          'downloaded', default=0)

args = parser.parse_args()

#image_folder = 'img_2016'
#data_path = 'flickr_data_01_2016.csv'

Path(args.image_folder).mkdir(parents=True, exist_ok=True)
data = pd.read_csv(args.data_path)

for _, post in tqdm(data.iloc[args.start_index:].iterrows(), total=data.shape[0] - args.start_index):
    url = f'https://live.staticflickr.com/{post.Server}/{post.ID}_{post.Secret}.jpg'
    try:
        urllib.request.urlretrieve(url, f'{args.image_folder}/{post.ID}.jpg')
    except HTTPError:
        print(url)


