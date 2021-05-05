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
    f = 255 * sigmoid(dot(w, x))
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

ar = .0003
ag = .0003
ab = .0003
Wr = [1,0,0,0,0,0,0,0,0]
Wg = [1,0,0,0,0,0,0,0,0]
Wb = [1,0,0,0,0,0,0,0,0]

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
        else:
            img[i, j] = [0,0,0]


# Train a fuck ton of times
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
        Wr[j] = Wr[j] - (ar*(model(tempWr, x) - yr)*x[j])
        Wg[j] = Wg[j] - (ag*(model(tempWg, x) - yg)*x[j])
        Wb[j] = Wb[j] - (ab*(model(tempWb, x) - yb)*x[j])

        
        # adjust learning rate
        if abs(model(tempWr, x) - yr) > 120:
            ar = 0.002
        elif abs(model(tempWr, x) - yr) > 30:
            ar = 0.0005
        elif abs(model(tempWr, x) - yr) > 0:
            ar = 0.0002

        if abs(model(tempWg, x) - yg) > 120:
            ag = 0.001
        elif abs(model(tempWg, x) - yg) > 30:
            ag = 0.0005
        elif abs(model(tempWg, x) - yg) > 0:
            ag = 0.0002

        if abs(model(tempWb, x) - yb) > 120:
            ab = 0.001
        elif abs(model(tempWb, x) - yb) > 30:
            ab = 0.0005
        elif abs(model(tempWb, x) - yb) > 0:
            ab = 0.0002
        '''
        if abs(model(tempWr, x) - yr) > 120:
            ar = 0.005
        elif abs(model(tempWr, x) - yr) > 60:
            ar = 0.0006
        elif abs(model(tempWr, x) - yr) > 0:
            ar = 0.0002

        if abs(model(tempWg, x) - yg) > 120:
            ag = 0.1
        elif abs(model(tempWg, x) - yg) > 60:
            ag = 0.0009
        elif abs(model(tempWg, x) - yg) > 0:
            ag = 0.0001

        if abs(model(tempWb, x) - yb) > 120:
            ab = 0.1
        elif abs(model(tempWb, x) - yb) > 30:
            ab = 0.001
        elif abs(model(tempWb, x) - yb) > 0:
            ab = 0.0001
        '''
            
        #print("Loss Red", model(tempWr, x), yr, model(tempWr, x) - yr)
        #print("Loss Green", model(tempWg, x), yg, model(tempWg, x) - yg)
        #print("Loss Blue", model(tempWb, x), yb, model(tempWb, x) - yb)
        #print()

#print(Wr,"\n")
#print(Wg,"\n")
#print(Wb,"\n")

print("Testing advanced agent")
# basic coloring agent
for i in range(rows):
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




    
