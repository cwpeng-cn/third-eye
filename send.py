import requests
import json
import base64
import os

PATH = "./temp"


def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


for name in os.listdir(PATH):
    path = os.path.join(PATH, name)
    img_str = getByte(path)
    url = 'http://43.129.251.135:8080/upload'
    data = {'picture': [{"filename": name, "content": img_str}]}
    json_mod = json.dumps(data)
    res = requests.post(url=url, data=json_mod)
