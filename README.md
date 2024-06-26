# Final-Project

# Goal: 
To train an object detection model to detect 4 classes using both our own dataset and found data

1. plastic bottles
2. glass bottles
3. Cans
4. Crushed Cans

# Where did we get these photos? 
We got photos from two sources, the first was two online data sets that had a huge number of photos of trash from various angles. However these had to be manually sorted through to find images that were suitable for training the model. After just using these images proved less accurate than we had hoped, we also decided to take a few hundered photos on photobooth to increase the size of the dataset to increase the accuracy. 

# What did did we  use to train the model?
We utilized the website, Roboflow, which makes training models understandable for almost all coding backgrounds

# Pretrained models we used to fine tune to our own dataset
1. Yolov5 (for our initial 2  models)
2. Yolov9 (ended up being much more accurate)

# How did we fine tune the model?
We used two google collab files "training_collab(Yolov5)" and "training_collab(Yolov9)"  These Google Collab were helpful in streamlining
the process and providing a GPU to utilize. 


# We ended up creating 3 python files to do the following:

1. Code to Detect Local Image: 
   Takes in a pathfile image and creates predictions based on the photo. If there are any predictions it will output the photo with a box and classify the object.
   We did this by overlaying a square and toxbox on the original photo.

3. TakePhoto:
   Opens your webcam. If you press s, it will take a screenshot and process the photo similar to how Code to Detect Local Image does.
   If youre webcam is frozen, we added a break feature where pressing q will stop the code from running.

4. Webcam:
   Opens your webcam. The idea was to run our object detection model on a livestream from the device's webcam. It runs similarly to
   Take Photo, but runs continously. Unofortunately, it was not successful due to the immense lag when the code ran. (The hardware running
   the object detection model may not be powerful enough to process the video frames in real-time.)

**Links:**

Data: https://drive.google.com/drive/folders/1ulHrOgEalxMRBXpuu1fimtriz-1ZN5r2?usp=sharing
file 'Images_combined' has all the images we ended up using for the analysis

Presentation: https://docs.google.com/presentation/d/1s9b17nYuEolI7k6D9eyt9NhuAL6pbmRW18QQQKoatBk/edit?usp=sharing
   

   

