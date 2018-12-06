#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:41:00 2018

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

#number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']

rootDir = "/Users/kxs/Desktop/check_code/test"
'''
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

    image = Image.open(captcha)
    image = np.array(image)
    return text, image

num = 1000
if __name__ == '__main__':
    for i in range(num):
        generateCaptchaTextandImage()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    print("finisheed!")
'''

image = []
def genList():
    #image = []
    for parent, dirnames, filenames in os.walk(rootDir):  
        for filename in filenames:  
            image.append(filename.replace(".jpg",""))
            # print("parent is:" + parent)
            # print("filename is:" + filename)
            # print("the full name of the file is:" + os.path.join(parent, filename))  
    return image

image = genList()

def getTestData():
    img = random.sample(image, 1)
    img = "".join(img)
    captchaImage = Image.open(rootDir + "/" + img + ".jpg")
    captchaImage = np.array(captchaImage)  
    return img, captchaImage

'''
def newGenerateCaptchaTextandImage():
    img = choice(image)
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
     if len(image.shape) > 2:
         r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]  
         gray = 0.299 * r + 0.587 * g + 0.114 * b
         #print(gray.shape)
     f = plt.figure()
     ax = f.add_subplot(111)
     ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
     plt.imshow(gray)
     plt.show()
