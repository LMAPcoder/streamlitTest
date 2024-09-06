import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

st.title("Title: Depth estimation")

ufile = st.file_uploader(
    "Upload image",
    type=['png', 'jpg']
)

if ufile:

    img = Image.open(ufile).convert("RGB")

    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type, force_reload=True)

    #midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    #transform = midas_transforms.small_transform

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 (or any other size)
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                             std=[0.229, 0.224, 0.225])
    ])

    midas.eval()

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        prediction = midas(img_tensor)

        output = prediction.detach().permute(1, 2, 0).numpy()

    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)


    col1, col2 = st.columns(2, gap="medium")

    with col1:
        new_size = (224, 224)
        resized_img = img.resize(new_size)
        st.image(resized_img, caption='Uploaded Image', use_column_width=True)

    with col2:

        st.pyplot(
                    fig=plt.imshow(output, cmap=cmap).get_figure(),
                    clear_figure=True,
                    use_container_width=True
                )