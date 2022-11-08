from flask import Flask, request, Response
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
from mnist_model import CustomNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
app = Flask(__name__)


@app.route("/predict", methods = ['POST'])
def save_image():
    if request.method == 'POST':
        photo = request.files["photo"]
        photo_str = photo.read()
        p_image = process_image(photo_str)
        label = inference(p_image)

        print(type(photo))
        if not os.path.exists(label):
            os.makedirs(label)
            photo.save("./"+label+"/picture1.jpg")
        else:
            count = len(os.listdir(label)) + 1
            photo.save("./"+label+"/picture"+str(count)+".jpg")
            
        
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

def process_image(photo_str):
    # photo_str = photo.read()
    photo_bytes = np.frombuffer(photo_str, np.uint8)
    
    image = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)
    thresh = cv2.adaptiveThreshold(grey.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 381, 10)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # preprocessed_digits = []
    # for each contour, add the bounded image to the list
    max = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        if h*w > max:
            max = h*w
            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y:y+h, x:x+w]
            
            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18,18))
            
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Adding the preprocessed digit to the list of preprocessed digits
        # preprocessed_digits.append(padded_digit)
    plt.imshow(digit, cmap="gray")
    plt.show()
    return padded_digit

def inference(p_image):

    plt.imshow(p_image, cmap="gray")
    plt.show()
    trans = transforms.ToTensor()
    # p_image = p_image.reshape([28,28])
    p_image_tensor = trans(p_image)
    p_image_tensor = torch.unsqueeze(p_image_tensor, 0)
    p_image_tensor = p_image_tensor.to(device) 
    
    model.eval()

    out = model(p_image_tensor)
    out_label = torch.argmax(out, dim = 1)
    print("Predicted label: ", out_label)
    return str(int(out_label[0]))


if __name__ == "__main__":
    device = torch.device('cpu')
    model = CustomNet()
    model.load_state_dict(torch.load("./mnist_model.pth"))
    model.to(device)
    app.run(host='0.0.0.0', port=5000)