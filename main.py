import numpy as np
import cv2
import matplotlib.pyplot as plt
from letter_recognizer import DataLoader, LetterRecognizer
import character_separator as character_separator



# initialize the model and dataset
LetterRecognizer = LetterRecognizer()
DataLoader = DataLoader()

# uncomment to train the model
#  LetterRecognizer.fit(DataLoader.X_train, DataLoader.Y_train)

# uncomment to evaluate the model
#  print('Evaluating the model...')
#  LetterRecognizer.evaluate(DataLoader.X_test, DataLoader.Y_test)

def predict_characters(img_path):
    img = cv2.imread(img_path)
    # separated characters from the text
    chars_arr = character_separator.separate_characters(img_path, (28,28))
    print('Making prediction...')
    for i in chars_arr:
        # make a prediction
        prediction = LetterRecognizer.predict(i[0], DataLoader.alphabets)
        print(prediction)

        # draw a bounding rectangle around the letter
        x,y,w,h = i[1]
        pt1 = (x,y)
        pt2 = (x+w, y+h)
        green_color = (0,255,0)
        thickness = 1

        cv2.rectangle(img, pt1, pt2, green_color, thickness)

        # draw the prediction of that letter on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x+2,y-5)
        fontScale = 1
        red_color = (0,0,255)
        lineType = 2

        cv2.putText(img,prediction, bottomLeftCornerOfText,
            font,
            fontScale,
            red_color,
            lineType)


    # display the final result
    #  cv2.imwrite('./result.png', img)
    cv2.imshow("Image" , img)
    cv2.waitKey(0)


#define path to an image 
img_path = './images/3.png'
predict_characters(img_path)


