import os
import numpy
from keras.models import load_model
from os.path import isdir, isfile, join
from os import listdir
import cv2

classifier = load_model("flower.h5")

def draw_test(nm, img, pred, original,ls):
    flower = ls[pred[0]]
    BLACK =[0,0,0]
    expanded_image = cv2.copyMakeBorder(img,80,0,0,100,cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, "Predicted-"+flower, (20,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(expanded_image, "Actual-"+original, (20,120), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow(nm,expanded_image)


def getRandomImage(path):
    folders = list(filter(lambda x: isdir(join(path,x)), os.listdir(path)))
    random_folder_no = numpy.random.randint(0,len(folders))
    random_folder = folders[random_folder_no]
    file_path = path+random_folder
    files = [f for f in listdir(file_path) if isfile(join(file_path,f))]
    random_file_no = numpy.random.randint(0,len(files))
    random_img = files[random_file_no]
    return cv2.imread(file_path+"/"+random_img), random_folder,folders


for i in range(0,15):
    input_im,fd,folder_list = getRandomImage("./17_flowers/validation/")
    input_original = input_im.copy()

    input_im = cv2.resize(input_im, (224,224), interpolation=cv2.INTER_LINEAR)
    input_im = input_im /255.
    input_im  = input_im.reshape(1,224,224,3)

    res = numpy.argmax(classifier.predict(input_im,1,verbose=0), axis=1)
    
    draw_test("Prediction", input_original,res,fd,folder_list)
    
    cv2.waitKey()

cv2.destroyAllWindows()
