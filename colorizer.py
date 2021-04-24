import cv2
import numpy
from PIL import Image
import random
from math import *

def kMeans(data, k):

    centers = []
    clusters = []
    similarCount = 0

    centers = [[0,0,0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]

    # pick random centers
    for i in range(k):
        #centers.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        clusters.append([])

    #print("clusters: ", clusters)
    prevCenters = centers.copy()

    #print(centers)
    # do k-means 100 times
    for i in range(100):

        # clear clusters
        for cluster in clusters:
            cluster = []
            
        # find clusters for current centers
        for color in data:
            distance = sqrt((256**2) + (256**2) + (256**2))
            index = 0
            group = 0

            # check which center is closest to curr data
            for center in centers:
                dist = sqrt((center[0] - color[0])**2 +
                            (center[1] - color[1])**2 +
                            (center[2] - color[2])**2)
                
                if dist < distance:
                    group = index
                    distance = dist
                
                index += 1

            clusters[group].append([color[0], color[1], color[2]])

        # get new centers for each cluster
        index = 0
        for cluster in clusters:
            size = len(cluster)
            sum_ = [0, 0, 0]

            if size == 0:
                return centers, False

            for value in cluster:
                sum_ = [sum_[0] + value[0], sum_[1] + value[1], sum_[2] + value[2]]
                
            mean = [sum_[0] / size, sum_[1] / size, sum_[2] / size]
            centers[index] = [round(mean[0]), round(mean[1]), round(mean[2])]
                
            index += 1

        sqrt((center[0] - color[0])**2 +
                            (center[1] - color[1])**2 +
                            (center[2] - color[2])**2)
        
        if prevCenters == centers:
            similarCount += 1
        else:
            similarCount = 0

        if similarCount >= 3:
            return centers, True

        prevCenters = centers.copy()
        #print(centers)
        
    return centers, True

def similarity(patch, comparision):
    diff = 0
    for i in range(9):
        diff += abs(patch[i] - comparision[i])

    diff = diff / 9
    return diff
        
    

''' MAIN '''
img = cv2.imread('pic2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img', img) 
#cv2.imshow('gray', simplified)
grayShape = gray.shape
rows = grayShape[0]
cols = grayShape[1]

inputs = []
outputs = []
print("Image Size: ", grayShape)


# get all colors
colordata = []
for i in range(rows):
    for j in range(cols):
        if i > 0 and i < rows-1 and j > 0 and j < cols-1:
            colordata.append(img[i, j])

# run k-means to reduce colors
'''
print("Running k-means")
aveColors, valid = kMeans(colordata, 5)
while not valid:
    aveColors, valid = kMeans(colordata, 5)
'''

# if using new image, comment this out and run k-means instead
aveColors = [[33, 85, 75], [207, 109, 68], [65, 122, 103], [121, 130, 128], [242, 180, 161]]
print("Average Colors: ", aveColors)

print("Training - Get inputs and outputs")
# get inputs and outputs
for i in range(rows):
    for j in range((int)(cols/2)):
        if i > 0 and i < rows-1 and j > 0 and j < cols-1:
            ''' GET INPUT '''
            # get raw 3x3 pixel values
            x_raw = [gray[i,j],
                 gray[i-1,j-1], gray[i-1,j], gray[i-1,j+1],
                 gray[i,j-1], gray[i,j+1],
                 gray[i+1,j-1], gray[i+1,j], gray[i+1,j+1]]

            # convert from 0/255 to -1/1
            x_processed = [ (n-128)/128 for n in x_raw]

            inputs.append(x_processed)

            ''' GET OUTPUT '''
            pixel = img[i, j]
            color = [0,0,0]
            index = 0
            
            distance = sqrt((256**2) + (256**2) + (256**2))
            for value in aveColors:
                dist = sqrt((value[0] - pixel[0])**2 +
                            (value[1] - pixel[1])**2 +
                            (value[2] - pixel[2])**2)
                
                if dist < distance:
                    color = aveColors[index]
                    distance = dist
                
                index += 1
                
            outputs.append(color)
            img[i, j] = color
        else:
            img[i, j] = [0,0,0]


print("Testing basic agent")
# basic coloring agent
for i in range(rows):
    
    if i == (int)(rows*.5):
        print("50% Complete")
    elif i == (int)(rows*.25):
        print("25% Complete")
    elif i == (int)(rows*.75):
        print("75% Complete")
    elif i == (int)(rows*.1):
        print("10% Complete")
    elif i == (int)(rows*.01):
        print("1% Complete")
                
    for j in range((int)(cols/2)+1, cols):
        if i > 0 and i < rows-1 and j > 0 and j < cols-1:
            ''' GET CURRENT PATCH '''
            # get raw 3x3 pixel values
            patch_raw = [gray[i,j],
                 gray[i-1,j-1], gray[i-1,j], gray[i-1,j+1],
                 gray[i,j-1], gray[i,j+1],
                 gray[i+1,j-1], gray[i+1,j], gray[i+1,j+1]]

            # convert from 0/255 to -1/1
            patch = [ (n-128)/128 for n in patch_raw]

            ''' COMPUTE LIKELY COLOR '''
            # order inputs from most to least similar to patch
            MostSimilarPatches = inputs.copy()
            MostSimilarPatches.sort(key=lambda element: similarity(patch, element))

            matchingColors = [[0, 0],[0, 1],[0, 2],[0, 3],[0, 4]]
            indexMostSimilar = 0
            for r in range(5):
                inputindex = inputs.index(MostSimilarPatches[r])
                outputcolor = outputs[inputindex]
                aveColorindex = aveColors.index(outputcolor)
                matchingColors[aveColorindex][0] += 1

                if r == 0:
                    indexMostSimilar = aveColorindex

            matchingColors.sort(key=lambda element: element)
            matchingColors.reverse()

            if not matchingColors[0][0] == matchingColors[1][0]: # if most common color found
                img[i, j] = aveColors[matchingColors[0][1]]
            else: # if tie
                img[i, j] = aveColors[indexMostSimilar]

        else:
            img[i, j] = [0,0,0]

#print(inputs)
#print(outputs)
print()
cv2.imshow('img', img)
