import streamlit as st
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

st.title("Title: Depth estimation")

ufile = st.file_uploader(
    "Upload image",
    type=['png', 'jpg']
)

if ufile:

    image = Image.open(ufile)

    img = np.array(image)

    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type, force_reload=True)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    midas.eval()

    #filename = 'dog.jpg'
    #img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    #plt.imshow(output, cmap=cmap)


    col1, col2 = st.columns(2, gap="medium")

    with col1:

        st.image(img, caption='Uploaded Image', use_column_width=True)

    with col2:

        st.pyplot(
                    fig=plt.imshow(output, cmap=cmap).get_figure(),
                    clear_figure=True,
                    use_container_width=True
                )