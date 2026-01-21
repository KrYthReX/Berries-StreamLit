# 🎈 Sitka Science Center Berry Phenological Analysis Tool

**https://sitka-berries.streamlit.app/**

This is a Streamlit app that allows a user to upload several images to be classified within 8 defined phenological stages. Pre-processing is available to the user to crop out a present black metadata bar and resize the image for space and time efficiency. Every image is then run through a selection of trained models including ResNet50, ViT, and a specially trained expert (MoE) model, generating probabilities for a speciic class label and a visualization to show how the model predicted. 

The models used (past and present) in this project can be downloaded on Hugginface at: **https://huggingface.co/andrewkota/Berries_ViT32_TestStreamlit/tree/main**
