# -*- coding: utf-8 -*-


from crypt import methods
from typing import overload
import streamlit as st
from PIL import Image
from skimage import io
import cv2 
import numpy as np
import argparse
import os




def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Masking','Image Colorization', 'Team')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Image Colorization':
        photo()
    if selected_box == 'Image Masking':
        photo2()
    if selected_box == "Team":
        team()
 
def load_image(filename):
    image = cv2.imread(filename)
    return image
    
def welcome():
    
    st.title('Image Restoration')
    
    st.subheader('DIGITAL RECREATORS')
    
    st.image('logo.jpeg',use_column_width=True)



 
def photo():
    

    st.title('Image Colourization')
    img_data = st.file_uploader(label='Drag and Drop Image',type =['png','jpg','jpeg'])
    
    if img_data is not None :
        
        
        uploaded_image = Image.open(img_data)

#save uploaded image locally
        uploaded_image.save("./tempimage/userinput.jpg")

#display image
        st.image(uploaded_image,caption = 'Your Image', use_column_width= True)

#model
        net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
        pts = np.load('./model/pts_in_hull.npy')

        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2,313,1,1)

        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

#read locally saved image
        image1 = io.imread("./tempimage/userinput.jpg")
            
        scaled = image1.astype("float32")/255.0
        lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab,(224,224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        
        ab = net.forward()[0, :, :, :].transpose((1,2,0))

        ab = cv2.resize(ab, (image1.shape[1],image1.shape[0]))

        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized,0,1)
        colorized = (255 * colorized).astype("uint8")

#delete local image for next image upload
        os.remove("./tempimage/userinput.jpg")

#display final image
        st.image(colorized,caption = 'Output', use_column_width= True)
    
        

def photo2():

    st.title('Image Inpainting')
    flags = cv2.INPAINT_TELEA
    #reading input image
    img_data2 = st.file_uploader(label='Drag and Drop Image',type =['png','jpg','jpeg'])
    #reading input mask
    
    
    if img_data2 is not None :
        uploaded_image2 = Image.open(img_data2)
        #reading input mask
        img_data3 = st.file_uploader(label='Drag and Drop Image Mask',type =['png','jpg','jpeg'])
        mask = Image.open(img_data3)
#save uploaded image locally
        uploaded_image2.save("./tempimage/userinput2.jpg")
        mask.save("./tempmask/usermask2.jpg")

#display image
        st.image(uploaded_image2,caption = 'Your Image', use_column_width= True)
        st.image(mask,caption = 'Your Image Mask', use_column_width= True)

#model
        flags = cv2.INPAINT_TELEA
        
        mask = cv2.cvtColor(uploaded_image2, cv2.COLOR_BGR2GRAY)
        output = cv2.inpaint(uploaded_image2, mask, 1, flags=flags)
        output = np.clip(output,0,1)
        output = (255 * output).astype("uint8")
        

#show output

        st.image(output, caption= 'Final image', use_column_width=True)

#remove temporary storage
        os.remove("./tempimage/userinput2.jpg")
        os.remove("./tempmask/usermask2.jpg")



        
       
        
        
def team():
    st.title('team page')
    st.subheader('DIGITAL RECREATORS')
    st.subheader('Megha Dhumal 19CE7015')
       
if __name__ == "__main__":
    main()
