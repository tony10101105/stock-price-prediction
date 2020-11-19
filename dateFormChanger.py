import csv
def changer1(year, month, day):
    if month == "Jan.":
        month = 1
    elif month == "Feb.":
        month = 2
    elif month == "March":
        month = 3
    elif month == "April":
        month = 4
    elif month == "May":
        month = 5
    elif month == "June":
        month = 6
    elif month == "July":
        month = 7
    elif month == "Aug.":
        month = 8
    elif month == "Sep.":
        month = 9
    elif month == "Oct.":
        month = 10
    elif month == "Nov.":
        month = 11
    elif month == "Dec.":
        month = 12
    
    return f'{year}-{month:02}-{day:02}'

def changer2(year, month, day):
    if month == "Jan":
        month = 1
    elif month == "Feb":
        month = 2
    elif month == "Mar":
        month = 3
    elif month == "Apr":
        month = 4
    elif month == "May":
        month = 5
    elif month == "Jun":
        month = 6
    elif month == "Jul":
        month = 7
    elif month == "Aug":
        month = 8
    elif month == "Sep":
        month = 9
    elif month == "Oct":
        month = 10
    elif month == "Nov":
        month = 11
    elif month == "Dec":
        month = 12
        
    if year == "10":
        year = 2010
    elif year == "11":
        year = 2011
    elif year == "12":
        year = 2012
    elif year == "13":
        year = 2013
    elif year == "14":
        year = 2014
    elif year == "15":
        year = 2015
    elif year == "16":
        year = 2016
    elif year == "17":
        year = 2017
    elif year == "18":
        year = 2018
    elif year == "19":
        year = 2019
    elif year == "20":
        year = 2020
        
    return f'{year}-{month:02}-{day:02}'

def changer3(year, month, day):
    year = int(year)
    month = int(month)
    day = int(day)
    return f'{year}-{month:02}-{day:02}'

def dateFormChanger(date):
    if type(date) == list:
        date = date[0]
    try:
        date = date.replace(',','')
        date = date.split(' ')
        return changer1(int(date[2]), date[0], int(date[1]))
    except:
        pass
    if type(date) == list:
        date = date[0]
    try:
        date = date.split('-')
        return changer2(date[2], date[1], int(date[0]))
    except:
        pass
    if type(date) == list:
        date = date[0]
    try:
        date = date.split('/')
        return changer3(date[0], date[1], date[2])
    except:
        raise Exception("date transform error!")
    

