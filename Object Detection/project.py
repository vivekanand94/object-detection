import cv2
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
classnames= []
classfile="coco.names"
with open(classfile,"rt") as f:
    classnames=f.read().rstrip("\n").split("\n")
configpath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightspath="frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(configpath,weightspath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True :
  success,img = cap.read()
  classIds, confs, bbox = net.detect(img,confThreshold=0.45)
  bbox=list(bbox)
  print(classIds,bbox,confs)

  

  if len(classIds) != 0:
    for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(255,165,0),thickness=2)
            cv2.putText(img,classnames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(255, 165, 0),2)
  cv2.imshow("output",img)
  cv2.waitKey(1)