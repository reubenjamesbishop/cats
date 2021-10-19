#importing the libraries
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import time

#loading the cat classifier model
cat_clf=joblib.load("Cat_Clf_model.pkl")

#functions to predict image
def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    
    return s

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute the probability of a cat being present in the picture
    
    Y_prediction = sigmoid((np.dot(w.T, X)+ b))
         
    return Y_prediction

# Designing the interface
st.title("Is it a Bird?")
st.write("Upload a picture to find out...")
# For newline
st.write('\n')

image = Image.open('images/bird.png')
show = st.image(image, use_column_width=True)

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    image = np.asarray(u_img)/255
    
    my_image= resize(image, (64,64)).reshape((1, 64*64*3)).T
    
if st.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            
            prediction = predict(cat_clf["w"], cat_clf["b"], my_image)
            time.sleep(2)
            st.success('Done!')
        
        #Formatted probability value to 3 decimal places
        probability = "{:.3f}".format(float(prediction*100))
        
        # Classify cat being present in the picture if prediction > 0.5
        
        if prediction > 0.5:
            
            st.header("It's a BIRD!" )
            
            st.write('**Probability: **',probability,'%')
                             
        else:
            st.header("It's not a bird...")
            
            st.write('**Probability: **',probability,'%')
    
    
    