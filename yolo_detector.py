import cv2
import numpy as np
# --------------- READ DNN MODEL ---------------
# Model configuration
config = "model/yolov3.cfg"
# Weights
weights = "model/yolov3.weights"
# Labels
LABELS = open("model/coco.names").read().split("\n")
#print(LABELS, len(LABELS))
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8") #visualizar un color para cada categoría
#print("colors.shape:", colors.shape)
# Load model
net = cv2.dnn.readNetFromDarknet(config, weights)
# --------------- READ THE IMAGE AND PREPROCESSING ---------------
cap = cv2.VideoCapture(0)
while True:
     ret, frame, = cap.read()
     if ret == False:
          break
     height, width, _ = frame.shape
    # Create a blob
     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), #input image, scale_factor,size 
                              swapRB=True, crop=False) #opencv lee en BGR y esta red lee en RGB, crop si la imagen se recortara despues de cambiar el tamaño
    #print("blob.shape:", blob.shape) #entrega 4 puntos
     # --------------- DETECTIONS AND PREDICTIONS ---------------
     ln = net.getLayerNames() #obtener los nombres de todas las capas de la red
     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] #encontrar las capas de salida 
     net.setInput(blob) #establecer bob como entrada de la red
     outputs = net.forward(ln) #propagación hacia delante    
     boxes = []
     confidences = []
     classIDs = []

     for output in outputs:  #Imprimir cada una de las detecciones
         for detection in output:
             scores = detection[5:] #80 valores de clases
             classID = np.argmax(scores) #indice de valor maximo de confianza
             confidence = scores[classID] #confianza más alta
             if confidence > 0.5:
                 box = detection[:4] * np.array([width, height, width, height]) #extraer los puntos donde se encuentra el objeto
                 #print("box:", box)
                 (x_center, y_center, w, h) = box.astype("int")
                 #print((x_center, y_center, w, h))
                 x = int(x_center - (w / 2))
                 y = int(y_center - (h / 2))
                 boxes.append([x, y, w, h])
                 confidences.append(float(confidence))
                 classIDs.append(classID)
     idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5) #tomar todas las detecciones y obtener un solo cuadro delimitador para cada objeto
     #print("idx:", idx)
     if len(idx) > 0: #primero aseguramos que exista un elemento en idx
         for i in idx: #ciclo para encerrar cada objeto
             (x, y) = (boxes[i][0], boxes[i][1]) #extraer puntos x, y
             (w, h) = (boxes[i][2], boxes[i][3])
             color = colors[classIDs[i]].tolist() #color con el que se encerrara el objeto 
             text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i]) #ubicar el nombre del objeto detectado y su valor de confianza
             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #cuadro delimitador
             cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, color, 2) #información del objeto con text
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()
