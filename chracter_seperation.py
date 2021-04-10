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