import cv2
# Importing the Opencv Lib,rary
import numpy as np
import pytesseract

# Importing NumPy,which is the fundamental package for scientific computing with Python

# Reading Imag 45695
img = cv2.imread("test1.jpg")
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
cv2.imshow("Original Image",img)


img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)

cv2.imshow("Gray Converted Image",img_gray)

noise_removal = cv2.bilateralFilter(img_gray,9,75,75)


cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)

cv2.imshow("Noise Removed Image",noise_removal)

equal_histogram = cv2.equalizeHist(noise_removal)
cv2.namedWindow("After Histogram equalisation",cv2.WINDOW_NORMAL)

cv2.imshow("After Histogram equalisation",equal_histogram)



kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

 

morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
cv2.namedWindow("Morphological opening",cv2.WINDOW_NORMAL)

cv2.imshow("Morphological opening",morph_image)


sub_morp_image = cv2.subtract(equal_histogram,morph_image)
cv2.namedWindow("Subtraction image", cv2.WINDOW_NORMAL)

cv2.imshow("Subtraction image", sub_morp_image)




ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)

cv2.imshow("Image after Thresholding",thresh_image)


canny_image = cv2.Canny(thresh_image,250,255)
cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)

cv2.imshow("Image after applying Canny",canny_image)

canny_image = cv2.convertScaleAbs(canny_image)


kernel = np.ones((3,3), np.uint8)

dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Dilation", dilated_image)
# Displaying Image

# Finding Contours in the image based on edges
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

 


screenCnt = None
count=0
idx=7
for c in contours:

 peri = cv2.arcLength(c, True)
 approx = cv2.approxPolyDP(c, 0.06 * peri, True) 

 if len(approx) == 4:  

  screenCnt = approx
  x, y, w, h = cv2.boundingRect(c)
  new_img = img[y:y + h,x:x + w]
  cv2.imwrite('cropped/' + str(idx) + '.jpg', new_img)
  idx+=1;
  break
final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

 

cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL)

cv2.imshow("Image with Selected Contour",final)



mask = np.zeros(img_gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)


y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))
y = cv2.equalizeHist(y)

final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)




