import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from imutils.contours import sort_contours
#  from letter_recognizer import DataLoader, LetterRecognizer

#Prcoessing an image 
def imageProcessing (path):
    img = cv2.imread (path)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert into grayscale
    blurredImage = cv2.GaussianBlur (grayImage, (5,5), 0)
    edgedImage = cv2.Canny(blurredImage, 30 ,150 )

    return edgedImage , grayImage


def findContours(processedImage, sort_method='left-to-right'):

    #Find and grab found contours from edged image
    contours = cv2.findContours(processedImage[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    #Sorting the result from top to bottom 
    contours = sort_contours(contours, method = sort_method)[0] 
    return contours

def separate_characters(path, img_size=(28,28)):
    #Initialize image 
    processedImage = imageProcessing(path)

    contours = findContours(processedImage, 'left-to-right')

    #Initialize an array to hold each character in the contours 
    chars = []

    #Start looping through every fond contours
    for cnt in contours:
        #Drawing a rectangle around each letter based on the contours
        x,y,w,h = cv2.boundingRect(cnt)
        
        #Filtering bounding boxes, removing small/big boxes
        if (w >= 5 and w <= 150) and (h >= 0 and h <= 200):

            #Computing region of interest 
            roi = processedImage[1][y:y + h, x:x + w]

            #Converting character's pixel to white with black background
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            #Extract width and height of the image 
            (tH, tW) = thresh.shape

            #Resize if width or hegith is too big
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            #Creating 50x50 bounding box
            (tH, tW) = thresh.shape     
            dX = int(max(0, 50 - tW) / 2.0)
            dY = int(max(0, 50- tH) / 2.0)

            #Creating  28x28 image for each character
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, img_size)

            #Uppdating list of characters   
            chars.append((padded, (x, y, w, h)))

    return chars
