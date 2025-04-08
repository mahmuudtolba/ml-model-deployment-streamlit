import streamlit as st
import boto3
import os
import time
from transformers import pipeline
import torch

BUCKET_NAME = 'mlops-udemy-course-aws'
MODEL_PATH  = 'model-dir'
s3_PREFIX = 'ml-models/tinybert-sentiment-analysis/'
BUCKET_NAME = 'mlops-udemy-course-aws'

DEVICE= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



s3 = boto3.client('s3')

# make title
st.title('Machine Learning Model Deployment At Streamlit Server')

# download button
download_button = st.button('Download Model')


def download_model(MODEL_PATH, s3_PREFIX):
    """
    Downloads model files from an S3 bucket to a local directory if they don't already exist.

    Parameters:
    - MODEL_PATH (str): Local path to store the model.
    - s3_prefix (str): S3 prefix where model files are located.

    If the model isn't already downloaded, this creates the target directory,
    fetches model files (filtered by 'tinybert-sentiment-analysis'), and stores them locally.
    Displays Streamlit toast notifications for status updates.
    """

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)
        paginator = s3.get_paginator('list_objects_v2')
        try:
            for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_PREFIX):
                if 'Contents' in result:
                    for key in result['Contents']:
                        s3_key = key['Key']
                        st.text(s3_key)
                        local_file = os.path.join(MODEL_PATH, os.path.relpath(s3_key, s3_PREFIX))
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                        s3.download_file(BUCKET_NAME, s3_key, local_file)

            st.toast("The Model is downloaded.")

        except Exception as e:
            st.error(f"Error downloading model: {e}")
    else:
        st.toast('The Model is already downloaded.')

    
def make_prediction(input_text):
    classifier = pipeline('text-classification', model='./model-dir', device=DEVICE)
    output_text = classifier(input_text)[0]
    st.text(f'{output_text['label']} with {round(output_text['score'] * 100 , 1 )}%')

if download_button:
    with st.spinner("Downloading... Please wait!"):
        download_model(MODEL_PATH ,s3_PREFIX)
        

# text area
input_text = st.text_area('Enter text' , placeholder = 'Type here.. ')



# download button
predict_button = st.button('Predict')

if predict_button:
   make_prediction(input_text)