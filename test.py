import pytesseract
from PIL import Image

for i in range(1, 7):
    #image = Image.open("license_plate_clarity_images/numberplate" + str(i) + "_license_plate.jpg")
    image = Image.open("test/" + str(i) + ".jpg")
    code = pytesseract.image_to_string(image, lang="eng", config="--psm 7")
    code = code.strip("\n")
    out_code = ""
    for n in range(len(code)):
        if code[n].isdigit() or code[n].isalpha():
            out_code = out_code + code[n].upper()
    print(out_code, end="|")

print()

print("3028BYS|ABC123|OTOBLOG|EWWDVID|H786POJ|4898GXY|")