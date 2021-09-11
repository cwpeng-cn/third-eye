import requests
import json
import base64

URL = 'http://43.129.251.135:8080/upload'
PATH = "./temp"


def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


def send(path):
    img_str = getByte(path)
    data = {'picture': [{"filename": path.split("/")[-1], "content": img_str}]}
    json_mod = json.dumps(data)
    requests.post(url=URL, data=json_mod)

# for name in os.listdir(PATH):
#     path = os.path.join(PATH, name)
#     send(path)
