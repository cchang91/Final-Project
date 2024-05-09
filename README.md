# Final-Project

#Goal: 
To create our own dataset and train an object detection model to detect 4 classes

1. plastic bottles
2. glass bottles
3. Cans
4. Crushed Cans

#Where did we get these photos: 
They were shot on Photobooth on our computer. We found it was best to have our dataset's photo 
quality mimic to where the code would be ran.

#What did did we  use to train the model?:
We utilized the website, Roboflow, which makes training models understandable for almost all coding backgrounds

#Pretrained models we used to fine tune to our own dataset
1. Yolov5 (for our initial 2  models)
2. Yolov9 (ended up being much more accurate)

#How did we fine tune the model?:
We used two the google collab files "training_collab(Yolov5)" and "training_collab(Yolov9)"  These Google Collab were helpful in streamlining
the process and providing a GPU to utilize. 


#We ended up creating 3 python files to do the following:

1. Code to Detect Local Image: 
Takes in a pathfile image and creates predictions based on the photo. If there are any predictions it will output the photo with  box and classify the object. (We did this by overlaying a square and toxbox on the original photo.

2. TakePhoto:
   Opens youre webcam. If you press s, it will take a screenshot and process the photo similar to how Code to Detect Local Image does.
   If youre webcam is forzen, we added a break feature where pressing q will stop the code from running.

3. Webcam:
   Opens youre webcam. The idea was to run our object detection model on a livestream from the device's webcam. It runs similarly to
   Take Photo, but runs continously. Unofrtunately, it was not successful due to the immense lag when the code ran. (The hardware running
   the object detection model may not be powerful enough to process the video frames in real-time.)

   

   

