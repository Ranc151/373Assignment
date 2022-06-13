import math
import sys
from pathlib import Path
import base64
import requests
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pytesseract
# import our basic, light-weight png reader library
from PIL import Image
import imageIO.png
# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
import extension
def computeAdaptiveThreshold(pixel_array, image_width, image_height, divisor):
    histogram = extension.computeHistogram(pixel_array, image_width, image_height)
    cumulativehistogram= extension.computeCumulativeHistogram(pixel_array, image_width, image_height)
    greyvalues = []
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] not in greyvalues:
                greyvalues.append(pixel_array[i][j])
    greyvalues.sort()
    qHq = []
    for i in range(len(greyvalues)):
        num = greyvalues[i]*histogram[i]
        qHq.append(num)

    theta = sum(qHq)/(image_width*image_height)
    left = 0
    right = 0
    index = 0
    for i in range(len(greyvalues)):
        if greyvalues[i] <= theta:
            left += qHq[i]
            index = i
        if greyvalues[i] > theta:
            right += qHq[i]

    result = left/cumulativehistogram[index] + right/(cumulativehistogram[-1]-cumulativehistogram[index])
    result = round(result*0.77)
    threshold = 0
    while threshold != result:
        threshold = result
        theta = result
        left = 0
        right = 0
        index = 0
        for i in range(len(greyvalues)):
            if greyvalues[i] <= theta:
                left += qHq[i]
                index = i
            if greyvalues[i] > theta:
                right += qHq[i]

        result = left / cumulativehistogram[index] + right / (cumulativehistogram[-1] - cumulativehistogram[index])
        result = round(result*divisor)
    return result
def main():
    for i in range(1,7):

        input_filename = "license_plate_clarity_images/numberplate"+str(i)+"_license_plate.jpg"
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = extension.readRGBImageToSeparatePixelArrays(input_filename)
        greyscale_pixel_array = extension.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
        threshold_value = computeAdaptiveThreshold(greyscale_pixel_array, image_width, image_height, 0.5)

        px_array = extension.computeThresholdGE(greyscale_pixel_array, threshold_value, image_width, image_height)
        pyplot.imshow(px_array, cmap='gray')
        pyplot.axis("off")
        pyplot.savefig('test/'+str(i)+'.jpg')
        #pyplot.show()


if __name__ == "__main__":
    main()