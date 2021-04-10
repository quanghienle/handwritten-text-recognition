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

