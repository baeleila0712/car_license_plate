import cv2
import pytesseract as pt
import matplotlib.pyplot as plt

# 讀取彩色的圖片
img = cv2.imread("02.jpg")
plt.imshow(img)
plt.title('Original')
plt.show()

#將圖片做些處理
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img1, (5,5), 10)
img3 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=1)
img4 = cv2.Canny(img3, 250, 100)
# plt.imshow(img4)
# plt.title('Canny')
# plt.show()

#二值化
i, img5 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY)

#取矩形,膨脹
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(43, 33))
img6 = cv2.dilate(img5,kernel)
# plt.imshow(img6)
# plt.title('Dilate')
# plt.show()


i, j = cv2.findContours(img6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result = None
for i1 in i:
    x,y,w,h = cv2.boundingRect(i1)
    if w > 2*h:
        print('success')
        plt.imshow(img[y:y+h,x:x+w])
        plt.title('Only LicensePlate')
        plt.show()
        result = img[y:y+h,x:x+w]

#搭配文字辨識看看
t = pt.image_to_string(result, "eng")
print("License Plate", t)

titles = ['Original', 'LicensePlate']  
images = [img, result]  
for i in range(2):  
   plt.subplot(2, 1, i+1), plt.imshow(images[i], 'gray')  
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()  

cv2.waitKey(0)
cv2.destroyAllWindows()