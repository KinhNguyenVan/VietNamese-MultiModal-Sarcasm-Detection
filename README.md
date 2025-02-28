# VietNamese-Multimodal-Detection
![Architecture]([image\edited_pipeline.png](https://github.com/KinhNguyenVan/VietNamese-MultiModal-Sarcasm-Detection/blob/main/image/edited_pipeline.png))
## Description
This project showcases a deep learning model designed to detect sarcasm in Vietnamese through a multimodal approach. Developed for an AI competition, the model processes input data consisting of images and their accompanying captions to classify four distinct labels: Text-sarcasm, Image-sarcasm, Not-sarcasm, and Multi-sarcasm.

Key Features
Utilizes ViSoBERT, a large language model for Vietnamese, to encode textual features.
Employs DeiT, a vision transformer, for image feature extraction.
Leverages cross-attention mechanisms to effectively combine features from captions, images, and OCR text.
Tackles the challenge of highly imbalanced training data, achieving an F1-score of 40.34%, compared to the top competition score of 46.58%.
Below is the architecture of the model:


This repository includes the code, data processing scripts, and details for reproducing the model's results.
