import cv2
from tracker import *



# Crear un objeto tipo Tracker. Atributos -> centroides
                                        # -> ID
tracker = EuclideanDistTracker()

video = cv2.VideoCapture("bloques.mp4")

# Sustractor que retorna un objeto con caracteristicas de una imagen sin fondo con parametros especificos para ser aplicados a un frame en especifico
object_detector = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=160,detectShadows=True)
#history: Number of last frames that affect the background model. Number of the last frame that are taken into consideretion
#varThreshold:  value used when computing the difference to extract the background. 
#               A lower threshold will find more differences with the advantage of a more noisy image.
while(1):
    _, frame = video.read() 
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[40: 1000,50: 1820] # Tamaño del video en donde los objetos pasaran. Video = 1080 x 1920

    # 1. Object Detection
    mask = object_detector.apply(roi)
    
    threshold_max = 254 #if a pixel value is greter than this, set to black. Otherwise it is white
    type_threshold = cv2.THRESH_BINARY
    output_value= 255

    # Pixel > threshold_max ? 
    # yes => pixel = 0
    # no  => pixel = output_value

    _, mask = cv2.threshold(mask, threshold_max, output_value, type_threshold)
    
    mode_param= cv2.RETR_TREE #Retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    method_param= cv2.CHAIN_APPROX_SIMPLE #compresses horizontal, vertical, and diagonal segments and leaves only their end points. 

    contours, _ = cv2.findContours(mask, mode_param, method_param)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 4500:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        image= roi
        text= "Objetos: {}".format(str(id))

        org=(10,100) #coordinates of the bottom - left corner of the text string in the image. The coordinates
                     #are represented as tuples of two values. i.e. (X,Y)
        font= cv2.FONT_HERSHEY_PLAIN
        fontScale=2
        color=(0,0,255)
        thickness_px=3
        
        cv2.putText(image,text , org, font, fontScale, color, thickness_px)
        
        start_point= (x,y) #It is the starting coordinates of rectangle. 
                           #The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        end_point = (x+w,y+h) #It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values 
                              #i.e. (X coordinate value, Y coordinate value).
        color = (0,255,0)  #BGR
        
        cv2.rectangle(image, start_point, end_point, color, thickness_px)
        # letrero con objetos encontrados
        

    cv2.imshow("Region de interes", roi)
    cv2.imshow("Video", frame)
    cv2.imshow("Mascara", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()