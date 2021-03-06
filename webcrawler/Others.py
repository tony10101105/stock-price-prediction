from bs4 import BeautifulSoup
import requests
import re
import csv
#import threading
#from time import sleep
#thread_list = []
headlines = {}
def work(url):
    try:
        text = requests.get(url).text
        bs = BeautifulSoup(text,"lxml")
        bs1 = bs.find("ul",{"class":"items hedSumm"})
        bs2 = bs1.find_all("h3",{"class":"headline"})
        bs3 = bs1.find_all("time")
        for i in range(len(bs2)):
            time = re.search(r"[A-Z][a-z]{2,4}\.? [0-9]{1,2}, [0-9]{4}",bs3[i].text).group()
            headline = bs2[i].find("a").text
            if time not in headlines:
                headlines[time] = [headline]
            else:
                if headline not in headlines[time]:
                    headlines[time].append(headline)
            #print(bs2[i].find("a").text)
            #print(re.search(r"[A-Z][a-z]{2,4}\.? [0-9]{1,2}, [0-9]{4}",bs3[i].text).group())
        #headlines = bs.find_all("li",{"xmlns":"http://www.w3.org/1999/html"})
        #print(headlines)
        
    except:
        print("some error!")
    """
    finally:
        sleep(0.05)
    """
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
urls = []
url1 = r"https://www.wsj.com/search/term.html?isAdvanced=true&articleType=Economy&daysback=4y&min-date=2009/01/01&max-date=2019/12/31&source=wsjarticle,press,newswire,wsjpro&page="
url2 = r"https://www.wsj.com/search/term.html?isAdvanced=true&articleType=Technology&daysback=4y&min-date=2009/01/01&max-date=2019/12/31&source=wsjarticle,press,newswire,wsjpro&page="
url3 = r"https://www.wsj.com/search/term.html?isAdvanced=true&articleType=Credit%20Markets&daysback=4y&min-date=2009/01/01&max-date=2019/12/31&source=wsjarticle,press,newswire,wsjpro&page="
url4 = r"https://www.wsj.com/search/term.html?isAdvanced=true&articleType=Politics%20and%20Policy&daysback=4y&min-date=2010/01/01&max-date=2020/01/01&source=wsjarticle,press,newswire,wsjpro&page="
search_dict = {
    "Economy":url1,
    "Tech":url2,
    "Credit Market":url3,
    "Politic":url4
}

for search_keyword in search_dict:
    url = search_dict[search_keyword]
    print(search_keyword)
    
    text = requests.get(url+"1").text
    bs = BeautifulSoup(text,"lxml")
    str1 = bs.find_all("li",{"class":"results-count"})[1].text
    all_num = ""
    for digit in str1:
        if digit.isdigit():
            all_num += digit
    all_num = int(all_num)
    print(all_num)
    
    for page in range(1,all_num+1):
        print(page)
        work(url+str(page))
        """
        thread = threading.Thread(target = work, args=(url+str(page),))
        thread_list.append(thread)
        if page%10 == 0:
            for work in thread_list:
                work.setDaemon(True)
                work.start()
                work.join()
            thread_list = []
        """

with open("output_others.csv","w",newline = "",encoding = "utf-8",errors="ignore") as f:
    writer = csv.writer(f)
    for time in headlines:
        for headline in headlines[time]:
            writer.writerow([time,headline])


#work(url+"55")
