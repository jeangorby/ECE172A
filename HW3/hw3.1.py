import cv2
import numpy as np
import matplotlib.pyplot as plt

def computeNormGrayHistogram(img):
    # For reference
    # histg = cv2.calcHist([img],[0],None,[32],[0,256])
    # plt.plot(histg)
    # plt.show()
    # return 0

    # Converting the image to grayscale
    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray forest", gray)
    # cv2.waitKey(0) 

    # Flatten the 2D array of images
    gray = np.array(gray)
    pixels  = gray.flatten()
    # print(pixels)

    # Initiliaze 32 length vector
    output = np.zeros(32)
    # Loop through all pixels and identify which bin they belong to
    for i in pixels:
        bin = i // (256/32)
        output[int(bin)] += 1

    height, width = img.shape[:2]

    output = output / (height*width)

    bins = np.arange(0,256,8)
    plt.bar(bins, output, width = 7, align = 'center')
    plt.title("Norm Gray Histogram")
    plt.show()

    # Return normalized vector
    return output 



def computeNormRGBHistogram(img):
    # Splitting the image into the respective channels
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    # Flatten the 2D array of images
    b = np.array(b)
    g = np.array(g)
    r = np.array(r)

    b_pixels  = b.flatten()
    g_pixels  = g.flatten()
    r_pixels  = r.flatten()

    # Initiliaze 32 length vector
    output_b = np.zeros(32)
    output_g = np.zeros(32)
    output_r = np.zeros(32)

    # Loop through all pixels and identify which bin they belong to
    for i in b_pixels:
        bin_b = i // (256/32)
        output_b[int(bin_b)] += 1

    for i in g_pixels:
        bin_g = i // (256/32)
        output_g[int(bin_g)] += 1

    for i in r_pixels:
        bin_r = i // (256/32)
        output_r[int(bin_r)] += 1

    height, width = img.shape[:2]

    output = np.hstack((output_b / (height*width), output_g / (height*width), output_r / (height*width)))
    
    
    bins_96 = np.arange(0,257*3,8)
    plt.bar(bins_96[0:31], output[0:31], width = 7, align = 'center', color = 'blue')
    plt.bar(bins_96[32:63], output[32:63], width = 7, align = 'center', color = 'green')
    plt.bar(bins_96[64:95], output[64:95], width = 7, align = 'center', color = 'red')
    plt.title("Norm RGB Histogram")
    plt.show()

    return output

def adaptiveHistogramEqualization(img, winSize):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = gray
    padded = np.pad(gray, (winSize//2, winSize//2), mode = 'symmetric')
    # cv2.imshow("mirrored", padded)
    # cv2.waitKey()

    # cv2.imshow("contextual region", window)
    # cv2.waitKey()

    for i in range(width):
        for j in range(height):
            pixel = gray.item(j, i)
            # new point on padded (i+winSize, j+winsize)
            window = padded[j: j + winSize - 1, i: i + winSize - 1]

            rank = window > pixel
            rank = np.sum(rank)
            output[j,i] = rank * (255/(winSize*winSize))

    return output



forest = cv2.imread("forest.jpg")
beach = cv2.imread("beach.png")
height, width = forest.shape[:2]
## Histogram of gray image
# forest_hist = computeNormGrayHistogram(forest)

## Histogram of rgb image
# forest_rgb_hist = computeNormRGBHistogram(forest)


## Histogram of flipped image
# forest_flipped = cv2.flip(forest, 1)

# forest_rgb_hist = computeNormRGBHistogram(forest)



## Doubling the R values
b = forest[:,:,0]
g = forest[:,:,1]
r = forest[:,:,2]
zeros = np.zeros_like(b)

# plt.imshow(cv2.merge([r, g, b]))
# plt.show()

# r_new = r.flatten()

# for index, value in enumerate(r_new):
#     if value*2 > 255:
#         r_new[index] = 255
#     else:
#         r_new[index] = value*2

# r_new = np.reshape(r_new, (height, width))

# forest_twice_red = cv2.merge([b, g, r_new])

# cv2.imshow("og forest", cv2.merge([b,g, r_new]))
# cv2.waitKey(0)

# forest_rgb_hist = computeNormRGBHistogram(forest_twice_red)


beach_ahe = adaptiveHistogramEqualization(beach, 129)

cv2.imshow("og beach", beach)
cv2.waitKey()
cv2.imshow("beach ahe", beach_ahe)
cv2.waitKey()

beach_hist = computeNormGrayHistogram(beach)

beach_ahe_hist = computeNormGrayHistogram(beach_ahe)
