from flask import Flask, request
import json
import os
import torch
from torchvision import transforms
from mnist_model import CustomNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
app = Flask(__name__)
UPLOAD_FOLDER = os.getcwd()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/predict", methods = ['POST'])
def save_image():
    if request.method == 'POST':
        photo = request.files["photo"]
        og_image, p_image = process_image(photo)
        label = inference(p_image)

        if not os.path.exists(label):
            os.makedirs(label)
            cv2.imwrite(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], label), 'picture1.jpg'), og_image)
        else:
            count = len(os.listdir(label)) + 1
            filename = "picture" + str(count) + ".jpg"
            cv2.imwrite(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], label), filename), og_image)
            
        
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

def process_image(photo):
    photo_str = photo.read()
    photo_bytes = np.frombuffer(photo_str, np.uint8)
    
    image = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
    wr_image = image.copy()

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)
    thresh = cv2.adaptiveThreshold(grey.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 381, 10)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the max contour (i.e the digit)
    max = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        if h*w > max:
            max = h*w
            digit = thresh[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (18,18))
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
    return wr_image, padded_digit

def inference(p_image):

    trans = transforms.ToTensor()
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