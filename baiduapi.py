import base64
import requests
from PIL import Image


for i in range(1,7):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=cz3vI9KtNDTiDYPjw5fedGcN&client_secret=WqLcW6gzXYYTh8YEIdhLuBPaq4GniEoG'
    response = requests.get(host)

    access_token = response.json()['access_token']
    request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/image_definition_enhance"


    im1 = Image.open(r'license_plate_images/numberplate'+str(i)+'_license_plate.png')
    im1 = im1.convert('RGB')
    im1.save(r'license_plate_images_jpg/numberplate'+str(i)+'_license_plate.jpg')
    f = open("license_plate_images_jpg/numberplate"+str(i)+"_license_plate.jpg","rb")
    img = base64.b64encode(f.read())

    params = {"image": img}

    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    imgdata = base64.b64decode(response.json()['image'])


    imgdata = base64.b64decode(response.json()['image'])

    file = open('after-'+str(i)+'.jpg', 'wb')
    file.write(imgdata)
    file.close()
