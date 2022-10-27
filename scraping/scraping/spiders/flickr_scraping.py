import csv
import scrapy
from datetime import datetime
import time
import json
import pandas as pd


class My_Scraper(scrapy.Spider):
    name = 'my_flickr_scraper'
    allowed_domains = ['flickr.com', 'staticflickr.com']
    savefile = 'flickr_data_2019.csv'
    unix_time_begin = time.mktime(datetime(2019, 1, 1, 0).timetuple())
    unix_time_end = time.mktime(datetime(2019, 12, 31, 23, 59).timetuple())
    cur_begin = unix_time_end - 86400
    cur_end = unix_time_end
    print(unix_time_begin, unix_time_end)
    page = 1
    url_template = 'https://api.flickr.com/services/rest/' \
                   '?method=flickr.photos.search' \
                   '&api_key=ed5468a328fabe0daaa6ba7fd989c2ac' \
                   '&tags=hamburg' \
                   '&media=photos' \
                   f'&page={page}' \
                   f'&min_upload_date={cur_begin}' \
                   f'&max_upload_date={cur_end}' \
                   '&format=json' \
                   '&per_page=500' \
                   '&extras=date_upload%2Cdescription%2Cgeo%2Ctags'
    data_csv = open(savefile, 'w+', newline='', encoding='utf-8-sig')
    csvwriter_data = csv.writer(data_csv)
    csvwriter_data.writerow(['ID', 'Owner', 'Secret', 'Server', 'Title', 'Description', 'Year', 'Month', 'Day',
                             'Time', 'Hashtags', 'Latitude', 'Longitude'])
    data_csv.close()

    def my_parse(self, response):
        total_pages = response.xpath('//photos/@pages').get()
        page = response.xpath('//photos/@page').get()
        print(total_pages)
        print(page)
        print(response.text)

    def generate_new_time_url(self):
        return 'https://api.flickr.com/services/rest/' \
               '?method=flickr.photos.search' \
               '&api_key=ed5468a328fabe0daaa6ba7fd989c2ac' \
               '&tags=hamburg' \
               '&media=photos' \
               f'&page={self.page}' \
               f'&min_upload_date={self.cur_begin}' \
               f'&max_upload_date={self.cur_end}' \
               '&format=json' \
               '&per_page=500' \
               '&extras=date_upload%2Cdescription%2Cgeo%2Ctags'

    def start_requests(self):
        yield scrapy.Request(self.url_template, callback=self.parse)

    def parse(self, response):
        data_csv = open(self.savefile, 'a', newline='', encoding='utf-8-sig')
        csvwriter_data = csv.writer(data_csv)
        result = response.body.decode('utf-8').split('jsonFlickrApi(')[1][:-1]
        result = json.loads(result)

        if len(result['photos']['photo']) == 0:
            self.log('Empty result, continue with next day')
            self.page = 1
            self.cur_end = self.cur_begin
            self.cur_begin = self.cur_end - 86400
            yield scrapy.Request(self.generate_new_time_url(), callback=self.parse)

        timestamp = result['photos']['photo'][-1]['dateupload']
        max_pages = result['photos']['pages']
        posts = result['photos']['photo']
        self.log(f'Saving {len(posts)} on page {self.page}/{max_pages} up to timestamp {timestamp}')
        for post in posts:
            row = []
            row.append(post['id'])
            row.append(post['owner'])
            row.append(post['secret'])
            row.append(post['server'])
            row.append(post['title'])
            row.append(post['description']['_content'])
            posted = datetime.utcfromtimestamp(int(post['dateupload']))
            row.append(posted.year)
            row.append(posted.month)
            row.append(posted.day)
            row.append(posted.time())
            row.append(post['tags'])
            row.append(post['latitude'])
            row.append(post['longitude'])
            # print(row)
            csvwriter_data.writerow(row)
        data_csv.close()
        """if len(result['photos']['photo']) == 0:
            self.log('Got empty result, query finished')
            return
        timestamp = int(str(response.body).split('dateupload')[-1].split(',')[0][3:-1])
        print(timestamp, (self.unix_time_begin < timestamp - 1))"""
        if self.page < max_pages: # continue with current day but next page
            self.page += 1
        else: # continue with next day
            self.page = 1
            self.cur_end = self.cur_begin
            self.cur_begin = self.cur_end - 86400
            if self.cur_begin < self.unix_time_begin:
                self.cur_begin = self.unix_time_begin
            if self.cur_end <= self.unix_time_begin:
                self.log('Reached begin of period to scrape')
                return
        yield scrapy.Request(self.generate_new_time_url(), callback=self.parse)

