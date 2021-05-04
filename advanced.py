import cv2
import numpy
from PIL import Image
import random
from math import *

'''
Input: x = (x0 ... x9)
Output: (r, g, b)
Model:
r(x) = 255sig(wR*x)
g(x) = 255sig(wG*x)
b(x) = 255sig(wB*x)

Loss = Sum(r(x) - x)^2
'''

def sigmoid(z):
    if z < -600:
        z = -600
        
    return (1 / (1+exp(-z)))

def dot(a, b):
    dotproduct = 0

    for i in range(len(a)):
        dotproduct += dotproduct + (a[i] * b[i])
        i += 1

    return dotproduct

def model(w, x):
    f = round(255 * sigmoid(dot(w, x)))
    return f


''' MAIN '''
img = cv2.imread('pic3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayShape = gray.shape
rows = grayShape[0]
cols = grayShape[1]

inputs = []
outred = []
outgreen = []
outblue = []
print("Image Size: ", grayShape)

a = 0.05
Wr = [1,.1,.1,.1,.1,.1,.1,.1,.1]
Wg = [1,.1,.1,.1,.1,.1,.1,.1,.1]
Wb = [1,.1,.1,.1,.1,.1,.1,.1,.1]

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
            #outputs.append(pixel)
            outred.append(pixel[0])
            outgreen.append(pixel[1])
            outblue.append(pixel[2])

            '''
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
            '''
        else:
            img[i, j] = [0,0,0]


# Train 100 times
for k in range(10000000):
    #randR = random.randint(0, 8)
    #randG = random.randint(0, 8)
    #randB = random.randint(0, 8)

    i = random.randint(0, len(inputs)-1)
    x = inputs[i]
    yr = outred[i]
    yg = outgreen[i]
    yb = outblue[i]

    tempWr = Wr.copy()
    tempWg = Wg.copy()
    tempWb = Wb.copy()
    for j in range(len(Wr)):
        #print(a*2*(model(tempWr, x) - yr)*x)
        Wr[j] = Wr[j] - (a*2*(model(tempWr, x) - yr)*x[j])
        Wg[j] = Wg[j] - (a*2*(model(tempWg, x) - yg)*x[j])
        Wb[j] = Wb[j] - (a*2*(model(tempWb, x) - yb)*x[j])

print(Wr)
print(Wg)
print(Wb)

print("Testing basic agent")
# basic coloring agent
for i in range(rows):
    '''
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
        '''        
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

            ''' COMPUTE COLOR '''
            r = model(Wr, patch)
            g = model(Wg, patch)
            b = model(Wb, patch)
            img[i, j] = [r, g, b]

        else:
            img[i, j] = [0,0,0]

#print(inputs)
#print(outputs)
print()
cv2.imshow('img', img)




    
