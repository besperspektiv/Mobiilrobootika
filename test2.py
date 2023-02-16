from PIL import Image
from operator import itemgetter
from PIL import Image, ImageOps
import numpy
import cv2

im = Image.open("captcha.png")

im2 = ImageOps.grayscale(im)
im2.save("output_grayscale.gif")
im = im2.convert("P")
his = im.histogram()
im2 = Image.new("P",im.size,255)




values = {}

for i in range(256):
    values[i] = his[i]

for j,k in sorted(values.items(), key=itemgetter(1), reverse=True)[:10]:
    print(j,k)

temp = {}

for x in range(im.size[1]):
    for y in range(im.size[0]):
        pix = im.getpixel((y,x))
        temp[pix] = pix
        if pix == 35: # these are the numbers to get_
            im2.putpixel((y,x),0)

open_cv_image = numpy.array(im2)
open_cv_image = open_cv_image[:, ::-1].copy()
im2.save("output.gif")

while True:
    cv2.imshow('graycsale image', open_cv_image)

