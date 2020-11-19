import os
import csv
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import datetime as DT
from dateFormChanger import dateFormChanger, changer1, changer2

def normalization(x):
    mean = np.mean(x)
    std = np.std(x)
    
    for i in range(len(x)):
        x[i] = (x[i] - mean) / std
        
    return x

class DataSet(Dataset):

    def __init__(self, sliding_window = [], mode = 'train'):

        ##preprocessing of numerical data
        self.closePrice = []
        self.dates = []
        self.slicedClosePrice = []
        self.slicedDates = []
        self.nextDayPrices = []

        if mode == 'train':
            STOCK_DATA_DIRECTORY = "./datasets/google_train.csv"
            NEWS_DATA_DIRECTORY = "./datasets/google_news_vector_train.csv"
        elif mode == 'test':
            STOCK_DATA_DIRECTORY = "./datasets/SP500_test.csv"
            NEWS_DATA_DIRECTORY = "./datasets/news_vector_test.csv"
        
        csvfileStock = open(STOCK_DATA_DIRECTORY)
        stockData = csv.reader(csvfileStock)
        
        for row in stockData:
            row[0] = row[0].replace('\ufeff', '')
            row[0] = dateFormChanger(row[0])
            self.closePrice.append(float(row[4]))
            self.dates.append(row[0])

        self.closePrice = normalization(self.closePrice)

        for sliceLength in sliding_window:
            for i in range(len(self.closePrice) - sliceLength + 1 -1 ):#last sliding window regarded
                self.slicedClosePrice.append(self.closePrice[i:sliceLength+i])
                self.slicedDates.append(self.dates[i:sliceLength+i])
                self.nextDayPrices.append(self.closePrice[sliceLength+i])

        ##preprocessing of textual data
        self.news = {}
        csvfileNews = open(NEWS_DATA_DIRECTORY)
        newsData = csv.reader(csvfileNews)

        for row in newsData:
            row[0] = row[0].replace('\ufeff', '')
            row[0] = dateFormChanger(row[0])
            vector = row[2:]
            for i in range(len(vector)):
                vector[i] = float(vector[i])
            self.news[row[0]] = vector

        self.news_vector = []
        for daysWindow in self.slicedDates:
            currentDate = daysWindow[-1].split('-')
            currentDate = DT.date(int(currentDate[0]), int(currentDate[1]), int(currentDate[2]))
            thirty_days_news = []
            
            for day in range(0, 30):
                key = str(currentDate-DT.timedelta(days = day))
                thirty_days_news.append(self.news[key])
                
            self.news_vector.append(thirty_days_news)
        
    def __getitem__(self, index):
        price = self.slicedClosePrice[index]
        date = self.slicedDates[index]
        news_vec = self.news_vector[index]
        nextPrice = self.nextDayPrices[index]
        
        price = torch.FloatTensor(price)
        news_vec = torch.FloatTensor(news_vec)
        
        return price, date, news_vec, nextPrice

    def __len__(self):
        assert len(self.slicedClosePrice) == len(self.slicedDates) == len(self.news_vector), "data length not matched"
        return len(self.slicedClosePrice)

