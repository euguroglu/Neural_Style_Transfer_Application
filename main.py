from flask import Flask, render_template,Response,request,jsonify
from flask import redirect,url_for
import os
from PIL import Image
import numpy as np
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import copy
import torch
import torch.optim as optim
from mains.utils import image_loader,ContentLoss,gram_matrix,StyleLoss,Normalization,get_style_model_and_losses
from mains.utils import get_input_optimizer,run_style_transfer,device,unloader


UPLOAD_FOLDER ='static/upload'


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/neural',methods=['GET','POST'])
def neural():
    if request.method == 'POST':
        f=request.files['image']
        f_2 = request.files['image2']
        filename = f.filename
        filename_2 = f_2.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        path2 = os.path.join(UPLOAD_FOLDER,filename_2)
        f.save(path)
        f_2.save(path2)
        # desired size of the output image
        style_img = image_loader("./static/upload/{}".format(filename))
        content_img = image_loader("./static/upload/{}".format(filename_2))
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        input_img = content_img.clone()
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img)
        save_img = output.squeeze(0)
        save_img = unloader(save_img)
        save_img.save("./static/predict/{}".format(filename_2))
        #prediction pass to pipeline model

        return render_template("neural.html",fileupload=True,img_name=filename_2)
    return render_template("neural.html",fileupload=False)



if __name__ == '__main__':

    app.run(debug=True)
