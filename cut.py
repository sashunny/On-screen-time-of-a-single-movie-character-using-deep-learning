import cv2

cap= cv2.VideoCapture('no_vinod.mp4')
i=0
 
image_folder = 'img_1'
while True:
    ret, frame = cap.read()
    
    if ret == False:
        break
    cv2.imwrite(image_folder+'/'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
