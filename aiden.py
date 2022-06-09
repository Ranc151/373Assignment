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

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate5.png"

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
    px_array = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    px_arr = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    px_arr = computeStandardDeviationImage5x5(px_arr, image_width, image_height)
    px_arr = scaleTo0And255AndQuantize(px_arr, image_width, image_height)
    px_arr = adaptiveThresholding(px_arr, image_width, image_height, 256)
    for n in range(5):
        px_arr = computeDilation8Nbh3x3FlatSE(px_arr, image_width, image_height)
    for n in range(5):
        px_arr = computeErosion8Nbh3x3FlatSE(px_arr, image_width, image_height)

    px_arr, ccsizes = computeConnectedComponentLabeling(px_arr, image_width, image_height)

    component_list = sorted([(key, value) for key, value in ccsizes.items()], key=lambda pair: pair[1], reverse=True)

    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    bbox_min_x = image_width
    bbox_max_x = 0
    bbox_min_y = image_height
    bbox_max_y = 0
    for i in range(len(component_list)):
        largest_component = component_list[i][0]
        for y in range(image_height):
            for x in range(image_width):
                if px_arr[y][x] == largest_component:
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
        if 1.5 <= aspect_ratio <= 5.0:
            break
        else:
            bbox_min_x = image_width
            bbox_max_x = 0
            bbox_min_y = image_height
            bbox_max_y = 0
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


def convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    greyscale = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            greyscale[i][j] = round(0.299 * px_array_r[i][j] + 0.587 * px_array_g[i][j] + 0.114 * px_array_b[i][j])
    return greyscale


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    min_val, max_val = (min([min(r) for r in pixel_array]), max([max(r) for r in pixel_array]))
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    if min_val == max_val:
        return result
    for i in range(image_height):
        for j in range(image_width):
            result[i][j] = round((pixel_array[i][j] - min_val) / (max_val - min_val) * 255)
    return result


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            a_list = [pixel_array[i + n][j + m] for n in range(-2, 3) for m in range(-2, 3)]
            mean = sum(a_list) / 25
            result[i][j] = round(math.sqrt(sum([(n - mean) * (n - mean) for n in a_list]) / 25))
    return result


def computeHistogram(pixel_array, image_width, image_height, nr_bins):
    result = [0.0] * nr_bins
    for i in range(image_height):
        for j in range(image_width):
            result[pixel_array[i][j]] += 1
    return result


def adaptiveThresholding(pixel_array, image_width, image_height, nr_bins):
    histogram = computeHistogram(pixel_array, image_width, image_height, nr_bins)
    j = 0
    n = image_width * image_height
    theta = round(sum([i * histogram[i] for i in range(nr_bins)]) / n)
    while True:
        n_obj = sum(histogram[:theta + 1])
        n_bg = sum(histogram[theta + 1:])
        avg_obj = sum([i * histogram[i] for i in range(0, theta + 1)]) / n_obj
        avg_bg = sum([i * histogram[i] for i in range(theta + 1, nr_bins)]) / n_bg
        new_theta = round(0.5 * (avg_obj + avg_bg))
        print("theta:", theta, "new_theta:", new_theta)
        if theta == new_theta:
            theta = new_theta + 80
            theta = 150
            break
        theta = new_theta

    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < theta:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            window = [[0] * 3 for x in range(3)]
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if i + m < 0 or j + n < 0 or i + m >= image_height or j + n >= image_width:
                        window[1 + m][1 + n] = 0
                    else:
                        window[1 + m][1 + n] = pixel_array[i + m][j + n]
            # check the window if current pixel fit
            fit = True
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if window[1 + m][1 + n] == 0:
                        result[i][j] = 0
                        fit = False
                        break
                else:
                    continue
                break
            if fit:
                result[i][j] = 1
    return result


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            window = [[0] * 3 for x in range(3)]
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if i + m < 0 or j + n < 0 or i + m >= image_height or j + n >= image_width:
                        window[1 + m][1 + n] = 0
                    else:
                        window[1 + m][1 + n] = pixel_array[i + m][j + n]
            # check the window if current pixel hit
            hit = False
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if window[1 + m][1 + n] == 1 or window[1 + m][1 + n] == 255:
                        result[i][j] = 1
                        hit = True
                        break
                else:
                    continue
                break
            if not hit:
                result[i][j] = 0
    return result


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height, initValue=False)
    label = 1
    ccsizes = {}
    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] == 1 or pixel_array[i][j] == 255) and not visited[i][j]:
                queue = Queue()
                queue.enqueue((i, j))
                visited[i][j] = True
                ccsizes[label] = 0
                while not queue.isEmpty():
                    ptr = queue.dequeue()
                    ccsizes[label] += 1
                    row = ptr[0]
                    col = ptr[1]
                    result[row][col] = label
                    if row - 1 >= 0:
                        if not visited[row - 1][col] and (
                                pixel_array[row - 1][col] == 1 or pixel_array[row - 1][col] == 255):
                            queue.enqueue((row - 1, col))
                            visited[row - 1][col] = True
                    if col - 1 >= 0:
                        if not visited[row][col - 1] and (
                                pixel_array[row][col - 1] == 1 or pixel_array[row][col - 1] == 255):
                            queue.enqueue((row, col - 1))
                            visited[row][col - 1] = True
                    if row + 1 < image_height:
                        if not visited[row + 1][col] and (
                                pixel_array[row + 1][col] == 1 or pixel_array[row + 1][col] == 255):
                            queue.enqueue((row + 1, col))
                            visited[row + 1][col] = True
                    if col + 1 < image_width:
                        if not visited[row][col + 1] and (
                                pixel_array[row][col + 1] == 1 or pixel_array[row][col + 1] == 255):
                            queue.enqueue((row, col + 1))
                            visited[row][col + 1] = True
                label += 1
    return result, ccsizes


if __name__ == "__main__":
    main()
