import requests
from lxml import etree
import pandas as pd
import re
import time
from snownlp import SnowNLP

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'referer': 'https://movie.douban.com/subject/1292052/reviews?start=20',
    'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
}

def get_data(params):
    response = requests.get('https://movie.douban.com/subject/20495023/reviews', params=params, headers=headers)
    if response.status_code == 200:
        content = etree.HTML(response.content.decode())
        print(f"Successfully retrieved page with start={params['start']}")
        return content
    else:
        print(f"Failed to retrieve page with start={params['start']}, status code: {response.status_code}")
        return None

def analysis_data(content):
    comment = content.xpath('//div[@class="short-content"]/text()')
    item_list = []
    for i in range(len(comment)):
        comment[i] = re.sub(r'\s+', ' ', comment[i])  
        comment[i] = comment[i].replace('\xa0', '') 
        comment[i] = comment[i].replace('(', '').replace(')', '')  
        comment[i] = comment[i].strip() 
        item_dic = dict(comment=comment[i])
        if item_dic['comment'] != '':
            item_list.append(item_dic)
    print('Finished analyzing list page')
    return item_list

def save_message(newcontent):
    df = pd.DataFrame(newcontent)
    df.to_excel('comments.xlsx', index=False)
    print('Saved successfully')
    return df

def main(max_page):
    all_comments = []
    for i in range(0, max_page * 20, 20):
        params = {'start': str(i)}
        content = get_data(params)
        if content is not None:
            newcontent = analysis_data(content)
            all_comments.extend(newcontent)
        else:
            print(f"Skipping page with start={i} due to retrieval failure.")
        time.sleep(1)  
    save_message(all_comments)

main(5)  

table = pd.read_excel('comments.xlsx')
table = pd.DataFrame(table, columns=['comment'])

def analyze_sentiment(text):
    s = SnowNLP(text)
    score = s.sentiments 
    return score

table['sentiment'] = table['comment'].apply(analyze_sentiment)

average_sentiment = table['sentiment'].mean()

print(table)
print(f"Average Sentiment Score: {average_sentiment}")

