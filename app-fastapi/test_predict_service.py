# --coding:utf-8--
import requests


data = {'data': '除傷害部分，非因宣判無罪，不得單獨據以聲請冤獄賠償外，其餘因擄人勒贖經不起訴處分及恐嚇危害安全經判決無罪確定，均非不得依********************************************'}

url_predict = 'http://0.0.0.0:8000/law_article_prediction'
print(requests.post(url_predict, json=data).text)
url_predict = 'http://0.0.0.0:8000/sentence_prediction'
print(requests.post(url_predict, json=data).text)