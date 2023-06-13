# %%
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from  threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ComputerVison(App):
    def build(self):
        
        
        
        self.window = GridLayout()
        self.window.cols = 2
        self.window.size_hint = (0.9, 0.9)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5 }
        
        # webcam
        self.web_cam = Image(size_hint=(1,.8))
        self.window.add_widget(self.web_cam)
        ## webcam with cv
        self.web_camcv = Image(size_hint=(1,.8))
        self.window.add_widget(self.web_camcv)

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1/33)
        

        #Logo
        #self.window.add_widget(Image(source = "Ghost.jpg", size_hint = (1,1)))

        # self.img1 = Image(size_hint = (1,1))
        # self.window.add_widget(self.img1)

        #ageRequest
        # self.ageRequest = Label(text = "    Insert your birth year", 
        # font_size = 20,
        # size_hint = (0.5, 0.5),
        # color = "#ffffff",
        # bold = True
        # )
        # self.window.add_widget(self.ageRequest)


        #year of birth

        # self.date = TextInput(multiline=False,
        # padding_y = (15, 15),
        # size_hint = (0.5, 0.5),
        # font_size = 20
        # )

        # self.window.add_widget(self.date)
  

        #button1

        # self.button = Button(text = "Calculate Age",
        # size_hint = (0.4, 0.4),
        # bold = True,
        # font_size = 20
        # )
        # self.button.bind(on_press = self.getAge)
        # self.window.add_widget(self.button)

        #button2
        self.button2 = Button(text = "Computer vision",
        size_hint = (0.1, 0.1),
        bold = True,
        font_size = 20
        )
        self.button2.bind(on_press = self.CV)
        self.window.add_widget(self.button2)

        #quit button
        self.quit = Button(text = "Quit",
        size_hint = (0.1, 0.1),
        # pos = (50,50),
        bold = True,
        font_size = 20
        )
        self.quit.bind(on_press = self.Quit)
        self.window.add_widget(self.quit)

        #setup video capture device

        self.capture = cv2.VideoCapture(0)

        return self.window
    # Run continuosly to get webcam feed
    def update(self, ev):

        # Read frame from opencv
        ret, frame = self.capture.read()
        #print(f"{len(frame)} update")

        # Flip horizontall and convert image to texture
        if frame is not None:
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture
        else:
            self.web_cam = Image()  
        
        return
    def getAge(self, event):
        today = 2023
        dob = self.date.text

        if dob.isdigit():
            age = int(today) - int(dob) 
            self.ageRequest.text = "You are " + str(int(age)) + " years old"
        else : 
            self.ageRequest.text = "Please insert it as an integer"
    def cv_thread(self,ev):
        Thread(target=self.CV).start()
    def CV(self,env):
        yolo = cv2.dnn.readNet( "./yolov3-tiny.weights","./yolov3-tiny.cfg")
        classes = []
        with open("./coco.names.txt","r") as f:
            classes = f.read().splitlines()
        # Capture frame-by-frame
        #self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        #print(f"{len(frame)} CV")
        # Perform computer vision on the frame
        #for i in range(10):
        image = frame
        blob = cv2.dnn.blobFromImage(image,1/255,(320,320),(0,0,0), swapRB = True, crop = False)
        yolo.setInput(blob)
        output_layer_name = yolo.getUnconnectedOutLayersNames()
        layeroutput = yolo.forward(output_layer_name)
        Width = image.shape[1]
        Height = image.shape[0]
        
        boxes = []
        confidences = []
        class_ids = []
        indexes = []
            
        for output in layeroutput:
            for detection in output:
                
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                #print(score[class_id])
                    
                if confidence > 0.4:
            
                    centre_X = int(detection[0]*Width)
                    centre_y = int(detection[1]*Height)

                    w = int(detection[2]*Width)
                    h = int(detection[3]*Height)
        
                    x = int(centre_X - h/2)
                    y = int(centre_y - w/2)
        
                    boxes.append([x,y,w,h])
                #print(boxes)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences ,0.5,0.6 )
            
            #print(indexes)
                    font = cv2.FONT_HERSHEY_PLAIN
                    colors = np.random.uniform(0, 255 , size = (len(boxes),3) )
        
        # drawing bounding boxes on the frame
                
            for i in indexes:
            
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i], 2))
                color = colors[i]

                cv2.rectangle(image,(x,y),(x+w, y+h), color, 2)
                cv2.putText(image, label +" " + confi, (x,y+20), font , 2, (255,255,255), 1)
            
        #cv2.imshow('Live', image)
            # Display the resulting frame

        if image is not None:

            buf2 = cv2.flip(image, 0).tobytes()
            print(f"image is not none {len(buf2)}") 
            imgcv_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            imgcv_texture.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')
            self.web_camcv.texture = imgcv_texture
        else:
             print("image is none")  
             self.web_camcv = Image(source = "Ghost.jpg", size_hint = (1,1))
            # Wait for key press
        #print("cv tamum shode")  
   
        
        #When everything done, release the capture
    def Quit(self, instance):
        App.get_running_app().stop()

if __name__ == "__main__":
    ComputerVison().run()











