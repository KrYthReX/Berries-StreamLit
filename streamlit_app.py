# streamlit run streamlit_app_modified.py

import streamlit as st
# import cv2
import numpy as np
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models_J import ViT, ResNet50, Expert_V1, SimpleCNN

st.set_page_config(layout="wide")

# #Precursor for data upload export
# if 'analysis_results' not in st.session_state:
#     st.session_state.analysis_results = []

# Berry Class Stages/Classes
PHENOLOGY_STAGES = [
    "breaking_leaf_buds", "increasing_leaf_size", "colored_leaves",
    "open_flowers", "ripe_fruits", "ripe_fruit_max",
    "pre_season", "post_season"
]

#File path/names for models that were uploaded to HuggingFace
REPO_ID = "andrewkota/Berries_ViT32_TestStreamlit"
# MODEL_FILENAME_ViT40_Katlian = "vit_bs32_ep40_katlian.pth"
# MODEL_FILENAME_ViT100_Katlian = "vit_bs42_ep100_katlian.pth"
MODEL_FILENAME_Exp_V1 = "exp_8stages_v1.pth"
# MODEL_FILENAME_ResNet100_FOB = "resnet50_bs42_ep100_fob.pth"

MODEL_FILENAME_ViT_ep120 = "vit_bs42_ep120_all_sites_95_5_focal1.5.pth"
MODEL_FILENAME_ResNet_ep120 = "resnet50_bs42_ep120_all_sites_95_5_focal1.5.pth"

MODEL_FILENAME_ConvNext = "convnext_base_bs42_ep120_all_sites_95_5.pth"

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

@st.cache_resource
def load_model(repo_id, filename, model_class):
    """
    Loads in trained mode in eval mode
    model_path to .pth file
    """
    # st.info(f"Downloading/loading model: {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.error("Please check your REPO_ID, MODEL_FILENAME, and that the file is public.")
        return None

    # st.info(f"File downloaded. Loading model from {model_path}...")

    try:
        model = model_class(num_classes=len(PHENOLOGY_STAGES))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        # st.success(f"Model {filename} has been loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading model from state_dict: {e}")
        # st.error("Test.")
        return None

def clear_uploads():
    st.session_state.uploader_key += 1

def preprocess_image(image_pil):
    """
    Converts an image to pytorch tensor
    """
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = test_transform(image_pil).unsqueeze(0)
    return image_tensor



def crop_metadata_bar(img_array):
    """
    Crops the image to a fixed height of 3192px from the top.
    """
    img_height = img_array.shape[0]
    target_height = 3192
    
    if img_height > target_height:
        st.success(f"Cropped image from {img_height}px to {target_height}px height.")
        return img_array[:target_height, :, :]
    else:
        st.info("Image height is already at or below target crop height. No crop needed.")
        return img_array



def resize_image(img_pil, base_width):
    """
    Resizes image according to new base_width (base 64) while maintaining originalo aspect ratio.
    """
    original_width, original_height = img_pil.size
    wpercent = (base_width / float(original_width))
    hsize = int((float(original_height) * float(wpercent)))
    img_resized = img_pil.resize((base_width, hsize), Image.Resampling.LANCZOS)
    st.success(f"Image resized to {base_width} x {hsize} from {original_width} x {original_height}.")
    return img_resized



MODELS_TO_LOAD = {
    "ViT (120 Epochs)": {
        "filename": MODEL_FILENAME_ViT_ep120,
        "class": ViT
    },
    # "ViT (40 Epochs)": {
    #     "filename": MODEL_FILENAME_ViT40_Katlian,
    #     "class": ViT
    # },
    "ResNet50 (120 Epochs)": {
        "filename": MODEL_FILENAME_ResNet_ep120,
        "class": ResNet50
    },
    "Expert": {
        "filename": MODEL_FILENAME_Exp_V1,
        "class": Expert_V1
    # }
    },
    "ConvNet":{
        "filename": MODEL_FILENAME_ConvNext,
        "class": SimpleCNN
    }
}

LOADED_MODELS = {}
# st.balloons("Loading models...")
for model_name, model_info in MODELS_TO_LOAD.items():
    if "YOUR_" in model_info["filename"]:
        st.warning(f"Skipping load for '{model_name}': Please update the placeholder filename.")
        LOADED_MODELS[model_name] = "Placeholder"
    else:
        try:
            model = load_model(REPO_ID, model_info["filename"], model_info["class"])
            if model:
                LOADED_MODELS[model_name] = model
            else:
                st.error(f"Failed to load '{model_name}'.")
                LOADED_MODELS[model_name] = "Failed"
        except Exception as e:
            st.error(f"An error occurred while loading '{model_name}': {e}")
            LOADED_MODELS[model_name] = "Failed"


st.title("🍓 Berry Image Analyzer")
st.info("This app analyzes multiple images. Set your processing and analysis settings *first*, then upload your images.")

st.header("1. Processing Settings")
st.caption("These settings will be applied to *all* uploaded images.")

st.subheader("Crop Out Metadata Bar")
st.caption("**Why?** The model was not trained on images containing the black metadata bar. We recommend cropping for best results if the bar is present to reduce model hallucinations.")
crop_image = st.checkbox("Crop black metadata bar")

st.subheader("Resize Image")
st.caption("**Why**? Resizing speeds up analysis as it reduces file size by up to 90%. We recommend *1024* as a good balance between size, quality, and clarity.")
resize_options = ["original", 64, 128, 256, 512, 1024, 2048, 4096]
selected_width = st.selectbox("Resize image (keep orig. aspect ratio):", options=resize_options)



st.header("2. Analysis Model")
st.caption("Choose the model(s) you want to run on all images.")

available_models = [name for name, model in LOADED_MODELS.items() if not isinstance(model, str)]
model_choices = None

if not available_models:
    st.error("No models were loaded. Check connection and HuggingFace repo.")
else:
    model_choices = st.multiselect(
        "Choose model(s):",
        available_models,
        default=available_models[:2] # Default to the first two models
    )

# st.header("3. Export Settings")
# st.caption("Settings for CSV export. Results are collected from image uploads, including metadata.")
# include_datetime = st.checkbox(
#     "Include date from image metadata (if available)", 
#     key="include_datetime"
# )

if st.button("Clear Results"):
    st.session_state.analysis_results = []
    st.toast("Stored analysis results cleared.")

st.header("3. Upload Images")
st.caption("Do not upload more than 50 images at a time. Doing so may cause instability in the app.")
st.caption("Note: While Streamlit won't accept folders, you can select several images for concurrent processing. We don't recommend selecting more than *50* images at this time (selecting more many cause instability).")
#Updated to upload multiple files. Won't accept folders, but will accept several images
# uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}" #Updated to fix ValueAssignmentNotAllowedError
)

st.button("Clear Uploaded Images", on_click=clear_uploads)

#Processing and analysis loop
if uploaded_files:
    if not model_choices:
        st.error("Please select at least one model from Step 2 before uploading.")
    else:
        
        st.success(f"Processing {len(uploaded_files)} images with {len(model_choices)} model(s)...")
        
        # Loop through each uploaded file
        for uploaded_file in uploaded_files:
                st.divider()
                st.subheader(f"Analyzing: {uploaded_file.name}")

                image_unprocessed = Image.open(uploaded_file)
                    
                    # Create columns for side-by-side view
                col1, col2 = st.columns(2)
                col1.image(image_unprocessed, caption="Original Image", width='content')


                image_to_display = image_unprocessed
                
                if crop_image:
                    image_array = np.array(image_unprocessed)
                    cropped_array = crop_metadata_bar(image_array)
                    image_to_display = Image.fromarray(cropped_array)

                if selected_width != "original":
                    base_width = int(selected_width)
                    image_to_display = resize_image(image_to_display, base_width)
                
                col2.image(image_to_display, caption="Processed Image", width='content')

                
                # 1. Preprocess the image
                image_tensor = preprocess_image(image_to_display)
                
                # 2. Create columns for the results, one for each model
                result_columns = st.columns(len(model_choices))

                # 3. Loop through each CHOSEN model and run analysis
                for i, model_name in enumerate(model_choices):
                    
                    # Get the column for this model
                    with result_columns[i]:
                        st.subheader(model_name) # Add header for clarity
                        model_to_run = LOADED_MODELS[model_name]

                        if model_to_run is None or isinstance(model_to_run, str):
                            st.error(f"Model '{model_name}' is not loaded.")
                            continue # Skip to the next model

                        with st.spinner(f"Model ({model_name}) is analyzing {uploaded_file.name}..."):
                            
                            # 2. Run inference
                            with torch.no_grad():
                                logits = model_to_run(image_tensor)
                                probabilities = torch.nn.functional.softmax(logits, dim=1)

                        # if model_name == "Expert":
                            
                            # 2a. Get MULTI-LABEL probabilities using sigmoid
                            probabilities = torch.sigmoid(logits)
                            probs_np = probabilities[0].numpy()
                            
                            # 3a. Get all predictions above a threshold
                            threshold = 0.5 # Make this a slider? Change threshold to .7?
                            predicted_indices = (probs_np > threshold).nonzero()[0]
                            
                            if len(predicted_indices) > 0:
                                predicted_labels = [PHENOLOGY_STAGES[i] for i in predicted_indices]
                                # Get the confidence of the top most prediction from this list
                                top_label_confidence = probs_np[predicted_indices].max()
                                display_value = ", ".join(predicted_labels)
                            else:
                                display_value = "No labels above threshold"
                                top_label_confidence = probs_np.max() # Show highest even if below thresh

                            # 4a. Display Metric (Multi-Label)
                            st.metric(
                                # label=f"Predictions (Threshold > {threshold*100}%)",
                                label = f"Predictions (Threshold > 50%)",
                                value=display_value,
                                delta=f"Top Confidence: {top_label_confidence:.2%}" if len(predicted_indices) > 0 else None,
                                delta_color="normal"
                            )
                            
                            top_class_index = probs_np.argmax()
                            top_class_name = PHENOLOGY_STAGES[top_class_index]
                            top_prob = probs_np[top_class_index]

                        
                        # else:
                        #     # 2b. Run original MULTI-CLASS logic
                        #     probabilities = torch.nn.functional.softmax(logits, dim=1)
                        #     probs_np = probabilities[0].numpy()

                        #     # 3b. Get top prediction
                        #     top_prob = probs_np.max()
                        #     top_class_index = probs_np.argmax()
                        #     top_class_name = PHENOLOGY_STAGES[top_class_index]

                        #     # 4b. Display Metric (Multi-Class)
                        #     st.metric(
                        #         label=f"Top Prediction",
                        #         value=f"{top_class_name}",
                        #         delta=f"Confidence: {top_prob:.2%}",
                        #         delta_color="normal"
                        #     )

                        # 5. Display Chart
                        # - Softmax: Bars will sum to 100% (multi-class classification)
                        # - Sigmoid: Bars will NOT sum to 100% (multi-label classification)
                        st.subheader("All Class Probabilities")
                        probs_df = pd.DataFrame(probs_np, index=PHENOLOGY_STAGES, columns=["Probability"])
                        st.bar_chart(probs_df)