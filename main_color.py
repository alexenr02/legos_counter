import cv2
import numpy as np
from tracker import *

#variables
org=(10,100) #coordinates of the bottom - left corner of the text string in the image. The coordinates
                #are represented as tuples of two values. i.e. (X,Y)
font= cv2.FONT_HERSHEY_PLAIN
fontScale=2
color_red=(0,0,255)
color_green = (0,255,0)  #BGR
color_blue = (255,0,0)
color_orange= (0,165,255)
thickness_px=3

# Crear un objeto tipo Tracker. Atributos -> centroides
                                        # -> ID
tracker_red = EuclideanDistTracker()
tracker_blue = EuclideanDistTracker()
tracker_green = EuclideanDistTracker()
tracker_orange = EuclideanDistTracker()
video = cv2.VideoCapture("bloques_medianos.mp4")

# Sustractor que retorna un objeto con caracteristicas de una imagen sin fondo con parametros especificos para ser aplicados a un frame en especifico
#history: Number of last frames that affect the background model. Number of the last frame that are taken into consideretion
#varThreshold:  value used when computing the difference to extract the background. 
#               A lower threshold will find more differences with the advantage of a more noisy image.

object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=120,detectShadows=True)

id_r=0
id_b=0
id_g=0
id_or=0
while(1):
    _, frame_total = video.read() 
    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value)
    # color space
    # Extract Region of interest
    frame = frame_total[40: 1000,50: 1820] # Tamaño del video en donde los objetos pasaran. Video = 1080 x 1920

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    height, width, _ = frame.shape

    # Set range for red color and 
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and 
    # define mask
    green_lower = np.array([36, 50, 70], np.uint8)
    green_upper = np.array([89, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Set range for blue color and
    # define mask
    orange_lower = np.array([10, 50, 70], np.uint8)
    orange_upper = np.array([24, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    res_red = cv2.bitwise_and(frame, frame, 
                              mask = red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    res_green = cv2.bitwise_and(frame, frame,
                                mask = green_mask)
      
    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(frame, frame,
                               mask = blue_mask)

    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernel)
    res_orange = cv2.bitwise_and(frame, frame,
                               mask = orange_mask)
    # Creating contour to track red color
    contours_red, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Creating contour to track blue color
    contours_blue, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Creating contour to track green color
    contours_green, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Creating contour to track green color
    contours_orange, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    detections_red = []
    for cnt in contours_red:
        # Calculate area and remove small elements
        area_r = cv2.contourArea(cnt)
        if area_r > 9000:
            x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
            detections_red.append([x_r, y_r, w_r, h_r])

    detections_blue = []
    for cnt2 in contours_blue:
        # Calculate area and remove small elements
        area_b = cv2.contourArea(cnt2)
        if area_b > 9000:
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt2)
            detections_blue.append([x_b, y_b, w_b, h_b])
    
    detections_green = []
    for cnt3 in contours_green:
        # Calculate area and remove small elements
        area_g = cv2.contourArea(cnt3)
        if area_g > 9000:
            x_g, y_g, w_g, h_g = cv2.boundingRect(cnt3)
            detections_green.append([x_g, y_g, w_g, h_g])

    detections_orange = []
    for cnt4 in contours_orange:
        # Calculate area and remove small elements
        area_or = cv2.contourArea(cnt4)
        if area_or > 9000:
            x_or, y_or, w_or, h_or = cv2.boundingRect(cnt4)
            detections_orange.append([x_or, y_or, w_or, h_or])

    # 2. Object Tracking
    boxes_ids_red = tracker_red.update(detections_red)
    boxes_ids_blue = tracker_blue.update(detections_blue)
    boxes_ids_green = tracker_green.update(detections_green)
    boxes_ids_orange = tracker_orange.update(detections_orange)


    for box_id_r in boxes_ids_red:
        x_r, y_r, w_r, h_r, id_r = box_id_r
        
        start_point= (x_r,y_r) #It is the starting coordinates of rectangle. 
                           #The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        end_point = (x_r+w_r,y_r+h_r) #It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values 
                              #i.e. (X coordinate value, Y coordinate value).
        cv2.rectangle(frame, start_point, end_point, color_red, thickness_px)
        # letrero con objetos encontrados
        
    text= "Objetos Rojos: {}".format(str(id_r))    
    cv2.putText(frame,text , org, font, fontScale, color_red, thickness_px)
    
    for box_id_b in boxes_ids_blue:
        x_b, y_b, w_b, h_b, id_b = box_id_b
        
        start_point= (x_b,y_b) #It is the starting coordinates of rectangle. 
                           #The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        end_point = (x_b+w_b,y_b+h_b) #It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values 
                              #i.e. (X coordinate value, Y coordinate value).
        cv2.rectangle(frame, start_point, end_point, color_blue, thickness_px)
        # letrero con objetos encontrados
        
    text= "Objetos azules: {}".format(str(id_b))    
    cv2.putText(frame,text , (10,200), font, fontScale, color_blue, thickness_px)
    
    for box_id_g in boxes_ids_green:
        x_g, y_g, w_g, h_g, id_g = box_id_g
        
        start_point= (x_g,y_g) #It is the starting coordinates of rectangle. 
                           #The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        end_point = (x_g+w_g,y_g+h_g) #It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values 
                              #i.e. (X coordinate value, Y coordinate value).
        cv2.rectangle(frame, start_point, end_point, color_green, thickness_px)
        # letrero con objetos encontrados
        
    text= "Objetos verdes: {}".format(str(id_g))    
    cv2.putText(frame,text , (10,300), font, fontScale, color_green, thickness_px)


    for box_id_or in boxes_ids_orange:
        x_or, y_or, w_or, h_or, id_or = box_id_or
        
        start_point= (x_or,y_or) #It is the starting coordinates of rectangle. 
                           #The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        end_point = (x_or+w_or,y_or+h_or) #It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values 
                              #i.e. (X coordinate value, Y coordinate value).
        cv2.rectangle(frame, start_point, end_point, color_orange, thickness_px)
        # letrero con objetos encontrados
        
    text= "Legos naranjas: {}".format(str(id_or))    
    cv2.putText(frame,text , (10,400), font, fontScale, color_orange, thickness_px)


    cv2.imshow("Region de interes", frame)
    cv2.imshow("Video", frame)
    cv2.imshow("green mask",green_mask)
    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()