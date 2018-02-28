import glob
import cv2
import math
import numpy as np
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import h5py
import scipy

from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import threshold_mean
from skimage import data
from skimage.filters import try_all_threshold
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color

from skimage import feature, io
from sklearn import preprocessing
import FeatureExtraction as Features




GrayImg = []
#Converting rgb to gray scale image
def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    grayImage = img

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage

#for i in range ( len( train_set_x_orig ) ):
 #   GrayImg.append( rgb_to_gray(train_set_x_orig[i]) )



FilteringImg = []
#Applying average filter to remove noise
#Gaussian Filtering    blur = cv2.GaussianBlur(img,(5,5),0)
def Filtering():
    for i in range ( len( train_set_x_orig ) ):
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(GrayImg[i],-1,kernel)
        FilteringImg.append(dst)
       # plt.subplot(121), plt.imshow(train_set_x_orig[0]), plt.title('Original')
       # plt.xticks([]), plt.yticks([])
       # plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
       # plt.xticks([]), plt.yticks([])
       # plt.show()
       # train_set_x_orig[i] = dst




GaussianFilterImg = []
# Applying average filter to remove noise
# Gaussian Filtering    blur = cv2.GaussianBlur(img,(5,5),0)
def GaussianFiltering():
    for i in range ( len( train_set_x_orig ) ):
        kernel = np.ones((5,5),np.float32)/25
        blur = cv2.GaussianBlur(GrayImg[i],-1,kernel)
        GaussianFilterImg.append(blur)




BinaryImage = []
#Thresholding an image
def Thresholding():
    for i in range ( len(train_set_x_orig) ):
        img = GrayImg[i]
        ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        BinaryImage.append(thresh1)



def Printing():
    for i in range ( len( train_set_x_orig ) ):
        print(i)
        plt.subplot(121), plt.imshow(train_set_x_orig[i]), plt.title('Filtered')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(BinaryImage[i]), plt.title('Binary')
        plt.xticks([]), plt.yticks([])
        plt.show()


def helperFunction(arr,index):
    sum = 0.0
    for i in range(4):
        sum = sum + arr.item(index, i)
    sum = sum/4.0
    return sum

arr = []
flag1 = 1
def trainImage(image_list):
    global  arr
    global flag1
    arr.clear()
    for index in range(len(image_list)):
        img = io.imread(image_list[index], as_grey=True)

        # path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/GRAY_IMAGE/'
        # path += str(index)
        # path += '.jpg'
        # if (flag1 == 0):
        #     cv2.imwrite(path, img)
        #
        infile = cv2.imread(image_list[index])
        infile = infile[:, :, 0]
        hues = (np.array(infile) / 255.) * 179
        outimageHSV = np.array([[[b, 255, 255] for b in a] for a in hues]).astype(int)
        outimageHSV = np.uint8(outimageHSV)
        # path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/HSV_IMAGE/'
        # path += str(index)
        # path += '.jpg'
        # if (flag1 == 0):
        #     cv2.imwrite(path, outimageHSV)
        outimageBGR = cv2.cvtColor(outimageHSV, cv2.COLOR_HSV2BGR)
        #
        # path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/BGR_IMAGE/'
        # path += str(index)
        # path += '.jpg'
        # if (flag1 == 0):
        #     cv2.imwrite(path, outimageBGR)



        rgb = io.imread(image_list[index])
        lab = color.rgb2lab(rgb)

        outimageBGR = lab

        for i in range(outimageBGR.shape[0]):
            for j in range(outimageBGR.shape[1]):
                sum = 0
                for k in range(outimageBGR.shape[2]):
                    sum = sum + outimageBGR[i][j][k]
                sum = sum / (3 * 255)
                if(i<img.shape[0] and j<img.shape[1]):
                    img[i][j] = sum

        S = preprocessing.MinMaxScaler((0, 19)).fit_transform(img).astype(int)
        Grauwertmatrix = feature.greycomatrix(S, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=20,
                                              symmetric=False, normed=True)

        arr.append(feature.greycoprops(Grauwertmatrix, 'contrast'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'correlation'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'homogeneity'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'ASM'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'energy'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'dissimilarity'))
        arr.append(Features.sumOfSquares(Grauwertmatrix))
        arr.append(Features.sumAverage(Grauwertmatrix))
        arr.append(Features.sumVariance(Grauwertmatrix))
        arr.append(Features.Entropy(Grauwertmatrix))
        arr.append(Features.Entropy(Grauwertmatrix))
        arr.append(Features.differenceVariance(Grauwertmatrix))
        arr.append(Features.differenceEntropy(Grauwertmatrix))
        arr.append(Features.informationMeasureOfCorelation1(Grauwertmatrix))
        arr.append(Features.informationMeasureOfCorelation2(Grauwertmatrix))
    flag1 = 1

arr1 = []

flag = 1
def testImage(image_list):
    global arr1
    global flag
    arr1.clear()
    for index in range(len(image_list)):
        img = io.imread(image_list[index], as_grey=True)

        infile = cv2.imread(image_list[index])
        infile = infile[:, :, 0]
        hues = (np.array(infile) / 255.) * 179
        outimageHSV = np.array([[[b, 255, 255] for b in a] for a in hues]).astype(int)
        outimageHSV = np.uint8(outimageHSV)

        # path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/HSV_IMAGE/'
        # path += 'p'
        # path += str(index)
        # path += '.jpg'
        # if (flag == 0):
        #     cv2.imwrite(path, outimageHSV)

        outimageBGR = cv2.cvtColor(outimageHSV, cv2.COLOR_HSV2BGR)

        # #######
        # path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/BGR_IMAGE/'
        # path += 'p'
        # path += str(index)
        # path += '.jpg'
        # if (flag == 0):
        #     cv2.imwrite(path, outimageBGR)

        # for i in range(outimageBGR.shape[0]):
        #     for j in range(outimageBGR.shape[1]):
        #         sum = 0;
        #         for k in range(outimageBGR.shape[2]):
        #             sum = sum + outimageBGR[i][j][k]
        #         sum = sum / (3 * 255)
        #         if (i < img.shape[0] and j < img.shape[1]):
        #             img[i][j] = sum

        rgb = io.imread(image_list[index])
        lab = color.rgb2lab(rgb)

        outimageBGR = lab

        for i in range(outimageBGR.shape[0]):
            for j in range(outimageBGR.shape[1]):
                sum = 0
                for k in range(outimageBGR.shape[2]):
                    sum = sum + outimageBGR[i][j][k]
                sum = sum / (3 * 255)
                if (i < img.shape[0] and j < img.shape[1]):
                    img[i][j] = sum

        path = 'C:/Users/Protap Chandra Ghose/Desktop/Database/GRAY_IMAGE/'
        path += 'p'
        path += str(index)
        path += '.jpg'
        if (flag == 0):
            cv2.imwrite(path, img)

        S = preprocessing.MinMaxScaler((0, 19)).fit_transform(img).astype(int)
        Grauwertmatrix = feature.greycomatrix(S, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=20,
                                              symmetric=False, normed=True)

        arr1.append(feature.greycoprops(Grauwertmatrix, 'contrast'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'correlation'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'homogeneity'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'ASM'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'energy'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'dissimilarity'))
        arr1.append(Features.sumOfSquares(Grauwertmatrix))
        arr1.append(Features.sumAverage(Grauwertmatrix))
        arr1.append(Features.sumVariance(Grauwertmatrix))
        arr1.append(Features.Entropy(Grauwertmatrix))
        arr1.append(Features.Entropy(Grauwertmatrix))
        arr1.append(Features.differenceVariance(Grauwertmatrix))
        arr1.append(Features.differenceEntropy(Grauwertmatrix))
        arr1.append(Features.informationMeasureOfCorelation1(Grauwertmatrix))
        arr1.append(Features.informationMeasureOfCorelation2(Grauwertmatrix))
    flag = 1

#print("After applying GLCM the features are : ")
def GLCM(Matrix,index,mask,flag):
    global  arr
    global  arr1
    if(flag==0):#training
        id = 0
        for i in range(3):
            for j in range(len(arr)):
                if( (mask & (1<<j)) == 0 ):
                    continue
                ret = helperFunction(arr[index*15+j],i)
                Matrix[id][index] = ret
                id = id + 1
    else :#testing
        id = 0
        for i in range(3):
            for j in range(len(arr)):
                if ((mask & (1 << j)) == 0):
                    continue
                ret = helperFunction(arr1[index*15+j], i)
                Matrix[id][index] = ret
                id = id + 1

    #Features.coRelation(Grauwertmatrix)
    #print(CorrelationtStats)


def pr():
    print('\n')
    print("Contrast                          : %f" % (np.mean(ContrastStats)))
    print("Energy                            : %f" % (np.mean(Energy)))
    print("Dissimilarity                     : %f" % (np.mean(Dissimilarity)))
    print("Correlation                       : %f" % (np.mean(CorrelationtStats)))
    print("Homogeneity                       : %f" % (np.mean(ASMStats)))
    print("Sum of Squares                    : %f" % (np.mean(Features.sumOfSquares(Grauwertmatrix))))
    print("Sum Average                       : %f" % (np.mean(Features.sumAverage(Grauwertmatrix))))
    print("Sum Variance                      : %f" % (np.mean(Features.sumVariance(Grauwertmatrix))))
    print("Sum Entropy                       : %f" % (np.mean(Features.sumEntropy(Grauwertmatrix))))
    print("Entropy                           : %f" % (np.mean(Features.Entropy(Grauwertmatrix))))
    print("Difference Variance               : %f" % (np.mean(Features.differenceVariance(Grauwertmatrix))))
    print("Difference Entropy                : %f" % (np.mean(Features.differenceEntropy(Grauwertmatrix))))
    print("Information measure of corelation1: %f" % (np.mean(Features.informationMeasureOfCorelation1(Grauwertmatrix))))
    print("Information measure of corelation2: %f" % (np.mean(Features.informationMeasureOfCorelation2(Grauwertmatrix))))





###parted - GOD work
#str.append( 'C:/Users/Protap Chandra Ghose/Desktop/Database/Brown/image18.jpg' )
#str.append( 'C:/Users/Protap Chandra Ghose/Desktop/Database/Brown/image2.jpg' )
#str.append( 'C:/Users/Protap Chandra Ghose/Desktop/Database/Brown/image15.jpg' )
#print("\nBrown")
#index=index+1
#GLCM()
#index=index+1
#print("\nNarrow")
#GLCM()
#index=index+1
#print("\nBlast")
#GLCM()
###End of GOD work






#img = cv2.imread('C:/Users/Protap Chandra Ghose/Desktop/Database/Brown/image4.jpg',0)
#ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#plt.imshow(thresh_img)
#plt.show()
#ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#for i in range(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()




#print("\nTotal Data : %d"% ( len(train_set_x_orig)+len(test_set_x_orig)) )
#print("Training Data : %d"% len(train_set_x_orig))
#print("Testing Data : %d"% len(test_set_x_orig))
#print("Cost after iteration %i: %f" % (i, cost))




#plt.imshow(train_set_x_orig[2])
#plt.show()

#plt.imshow(test_set_x_orig[0])
#plt.show()

#print("\n")

