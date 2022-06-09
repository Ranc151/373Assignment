import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# week 10
#Q3
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                pixel_array[i][j] = 255
            else:
                pixel_array[i][j] = 0
    return pixel_array


# Q4
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j])
    return greyscale_pixel_array

# Q5
def computeMinAndMaxValues(pixel_array):
    minimum = pixel_array[0][0]
    maximum = minimum
    for row in pixel_array:
        if min(row) < minimum:
            minimum = min(row)
        if max(row) > maximum:
            maximum = max(row)
    return minimum, maximum

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    result_list = createInitializedGreyscalePixelArray(image_width, image_height)
    minimum, maximum = computeMinAndMaxValues(pixel_array)
    if maximum == minimum:
        return result_list
    for i in range(image_height):
        for j in range(image_width):
            result_list[i][j] = round((pixel_array[i][j]-minimum)/(maximum-minimum)*255)
    return result_list

# Week 11
# Q6
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    out_list = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(2,image_height-2):
        for j in range(2, image_width-2):
            sum_list = []
            for n in range(-2,3):
                for m in range(-2,3):
                    sum_list.append(pixel_array[i+n][j+m])
            mean = sum(sum_list)/25
            avg_list = []
            for num in sum_list:
                avg_list.append((num-mean)* (num-mean))
            avg_sum = sum(avg_list)
            out_list[i][j] = round(math.sqrt(avg_sum/25))
    return out_list

# Week 12
# Q1
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    list_plus = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)
    out_list = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):
            list_plus[i][j] = pixel_array[i - 1][j - 1]

    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):

            sum_list = []
            for n in range(-1, 2):
                for m in range(-1, 2):
                    sum_list.append(list_plus[i + n][j + m])
            if 0 in sum_list:
                out_list[i - 1][j - 1] = 0
            else:
                out_list[i - 1][j - 1] = 1
    return out_list

# Q2
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    list_plus = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)
    out_list = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):
            list_plus[i][j] = pixel_array[i - 1][j - 1]

    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):

            sum_list = []
            for n in range(-1, 2):
                for m in range(-1, 2):
                    sum_list.append(list_plus[i + n][j + m])

            if sum(sum_list) != 0:
                out_list[i - 1][j - 1] = 1
            else:
                out_list[i - 1][j - 1] = 0
    return out_list

# Q3
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def bfs_traversal(pixel_array, visited, i, j, width, height, ccimg, count, q, x, y):
    number = 0

    q.enqueue((i, j))
    visited[i][j] = True
    while not q.isEmpty():
        a, b = q.dequeue()
        ccimg[a][b] = count
        number += 1
        for z in range(4):
            newI = a + x[z]
            newJ = b + y[z]
            if newI >= 0 and newI < height and newJ >= 0 and newJ < width and not visited[newI][newJ] and pixel_array[newI][newJ] != 0:
                visited[newI][newJ] = True
                q.enqueue((newI, newJ))
    return number

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    ccimg = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            visited[i][j] = False
    q = Queue()
    x = [-1, 0, 1, 0]
    y = [0, 1, 0, -1]
    ccsizedict = {}
    count = 1
    for i in range(image_height):
        for j in range(image_width):
            if not visited[i][j] and pixel_array[i][j] != 0:
                number = bfs_traversal(pixel_array, visited, i, j, image_width, image_height, ccimg, count, q, x, y)
                ccsizedict[count] = number
                count += 1
    return (ccimg, ccsizedict)


def computeHistogram(pixel_array, image_width, image_height):
    greyvalues = []
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] not in greyvalues:
                greyvalues.append(pixel_array[i][j])
    greyvalues.sort()
    histogram_dict={}
    for num in greyvalues:
        histogram_dict[num]=0
    for i in range(image_height):
        for j in range(image_width):
            histogram_dict[pixel_array[i][j]] += 1
    result_list = []
    for num in greyvalues:
        result_list.append(histogram_dict[num])

    return result_list


def computeCumulativeHistogram(pixel_array, image_width, image_height):
    greyvalues = []
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] not in greyvalues:
                greyvalues.append(pixel_array[i][j])
    greyvalues.sort()
    histogram_dict={}
    for num in greyvalues:
        histogram_dict[num]=0
    for i in range(image_height):
        for j in range(image_width):
            histogram_dict[pixel_array[i][j]] += 1
    result_list = []
    for num in greyvalues:
        result_list.append(histogram_dict[num])
    out_list = [result_list[0]]
    for i in range(1,len(result_list)):
        out_list.append(result_list[i] + out_list[-1])
    return out_list

def computeAdaptiveThreshold(pixel_array, image_width, image_height):
    histogram = computeHistogram(pixel_array, image_width, image_height)
    cumulativehistogram=computeCumulativeHistogram(pixel_array, image_width, image_height)
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
        result = round(result*0.77)
    return result


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True
    for i in range(1,7):
        # this is the default input image filename
        input_filename = "numberplate"+str(i)+".png"

        if command_line_arguments != []:
            input_filename = command_line_arguments[0]
            SHOW_DEBUG_FIGURES = False

        output_path = Path("output_images")
        if not output_path.exists():
            # create output directory
            output_path.mkdir(parents=True, exist_ok=True)

        output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
        if len(command_line_arguments) == 2:
            output_filename = Path(command_line_arguments[1])


        # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
        # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
        # setup the plots for intermediate results in a figure
        fig1, axs1 = pyplot.subplots(2, 2)
        axs1[0, 0].set_title('Input red channel of image')
        axs1[0, 0].imshow(px_array_r, cmap='gray')
        axs1[0, 1].set_title('Input green channel of image')
        axs1[0, 1].imshow(px_array_g, cmap='gray')
        axs1[1, 0].set_title('Input blue channel of image')
        axs1[1, 0].imshow(px_array_b, cmap='gray')


        # STUDENT IMPLEMENTATION here

        greyscale_pixel_array  = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
        smoothed_image  = computeStandardDeviationImage5x5(greyscale_pixel_array , image_width, image_height)
        contrast_stretched_pixel_array  = scaleTo0And255AndQuantize(smoothed_image , image_width, image_height)
        threshold_value = computeAdaptiveThreshold(contrast_stretched_pixel_array,image_width,image_height)
        px_array  = computeThresholdGE(contrast_stretched_pixel_array, threshold_value, image_width, image_height)
        for i in range(4):
            px_array = computeDilation8Nbh3x3FlatSE(px_array , image_width, image_height)
        for i in range(4):
            px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

        (ccimg, ccsizes) = computeConnectedComponentLabeling(px_array, image_width, image_height)

        component_list = sorted([(key, value) for key, value in ccsizes.items()], key=lambda pair: pair[1], reverse=True)
        # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
        bbox_min_x = image_width
        bbox_max_x = 0
        bbox_min_y = image_height
        bbox_max_y = 0
        aspect_ratio = 10.0
        index = 0
        while 1.5 >= aspect_ratio or aspect_ratio >= 5.0:
            bbox_min_x = image_width
            bbox_max_x = 0
            bbox_min_y = image_height
            bbox_max_y = 0
            largest_component = component_list[index][0]
            for y in range(image_height):
                for x in range(image_width):
                    if ccimg[y][x] == largest_component:
                        if y < bbox_min_y:
                            bbox_min_y = y
                        if y > bbox_max_y:
                            bbox_max_y = y
                        if x < bbox_min_x:
                            bbox_min_x = x
                        if x > bbox_max_x:
                            bbox_max_x = x
            box_width = bbox_max_x - bbox_min_x
            box_height = bbox_max_y - bbox_min_y
            aspect_ratio = box_width / box_height
            index += 1
            bbox_min_x -= 5
            bbox_min_y -= 5
            bbox_max_x += 5
            bbox_max_y += 5
        print(bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y)

        # Draw a bounding box as a rectangle into the input image
        axs1[1, 1].set_title('Final image of detection')
        axs1[1, 1].imshow(greyscale_pixel_array,cmap='gray')
        #axs1[1, 1].imshow(px_array, cmap='gray')
        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                         edgecolor='g', facecolor='none')
        axs1[1, 1].add_patch(rect)



        # write the output image into output_filename, using the matplotlib savefig method
        extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
        pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)


        if SHOW_DEBUG_FIGURES:
            # plot the current figure
            pyplot.show()


if __name__ == "__main__":
    main()