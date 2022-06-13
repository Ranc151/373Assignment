import base64

import requests
from PIL import Image

for i in range(1, 7):
    # extension part make image more clarity
    # change the png image to jpg image

    # get access to the api
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=cz3vI9KtNDTiDYPjw5fedGcN&client_secret=WqLcW6gzXYYTh8YEIdhLuBPaq4GniEoG'
    response = requests.get(host)
    access_token = response.json()['access_token']
    request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/contrast_enhance"
    # encode image to base64
    f = open("test/" + str(i) + ".jpg", "rb")
    img = base64.b64encode(f.read())
    # access api
    params = {"image": img}
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    imgdata = base64.b64decode(response.json()['image'])
    # get the response
    file = open("test/after" + str(i) + ".jpg", 'wb')
    file.write(imgdata)
    file.close()