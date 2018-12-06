#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:04:38 2018

@author: kxs
"""

from captcha.image import ImageCaptcha  
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
from random import choice
import sys

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']

rootDir = "/Users/kxs/Desktop/check_code/train"


# generate a random waiting line
def randomText(chars=number, size=4):
    text = []
    for i in range(size):
        c = random.choice(chars)
        text.append(c)
    return text



def generateCaptchaTextandImage():
    image = ImageCaptcha()

    text = randomText()
    text = ''.join(text)

    captcha = image.generate(text)
    image.write(text, text + '.jpg')  

    imageCaptcha = Image.open(captcha)
    imageCaptcha = np.array(image)
    return text, imageCaptcha

'''
num = 100
if __name__ == '__main__':
    for i in range(num):
        generateCaptchaTextandImage()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    print("finisheed!")
'''
def genList():
    imageList = []
    for parent, dirnames, filenames in os.walk(rootDir): 
        for filename in filenames:  
            imageList.append(filename.replace(".jpg",""))
            #print("parent is:" + parent)
            #print("filename is:" + filename)
            #print("the full name of the file is:" + os.path.join(parent, filename)) 
            #print("==========================")
    return imageList
imageList = genList()

#print(random.choice(imageList))
#print(imageList)
'''
for i in range(10):
    img = random.sample(imageList, 1)
    img = "".join(img)
    print(img)
'''
#print("".join(['02']))
def newGenerateCaptchaTextandImage():
    img = random.sample(imageList, 1)
    img = "".join(img)
    captchaImage = Image.open(rootDir + "/" + img + ".jpg")
    captchaImage = np.array(captchaImage)  
    return img, captchaImage

'''
if __name__ == '__main__':
     # test
     # text, image = genCaptchaTextandImage()
     #
     # f = plt.figure()
     # ax = f.add_subplot(111)
     # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
     # plt.imshow(image)
     # plt.show()
     #

     text, image = newGenerateCaptchaTextandImage()

     f = plt.figure()
     ax = f.add_subplot(111)
     ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
     plt.imshow(image)
     plt.show()
'''