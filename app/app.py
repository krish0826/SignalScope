import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SignalScope",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0d10;
        color: #f5f5f5;
    }

    /* Remove Streamlit top padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 820px;
    }

    /* Hide default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Logo */
    .logo {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .logo-dot {
        color: #7c8cff;
    }

    /* Subtitle */
    .subtitle {
        color: #8d939d;
        font-size: 15px;
        margin-bottom: 38px;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: #111419;
        border: 1px solid #252a32;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #111419;
        border: 1px dashed #343a44;
        border-radius: 12px;
    }

    /* Image */
    [data-testid="stImage"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Analyze button */
    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 12px;
        border: none;
        background: #f4f4f5;
        color: #090a0c;
        font-size: 15px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #ffffff;
        transform: translateY(-1px);
    }

    /* Result card */
    .result-card {
        background: #111419;
        border: 1px solid #252a32;
        border-radius: 18px;
        padding: 28px;
        margin-top: 25px;
    }

    .result-label {
        color: #858b95;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 20px;
    }

    .confidence {
        font-size: 14px;
        color: #9da3ad;
    }

    .confidence strong {
        color: #ffffff;
    }

    /* Probability */
    .prob-title {
        font-size: 13px;
        color: #969ca6;
        margin-top: 20px;
        margin-bottom: 7px;
    }

    /* Divider */
    hr {
        border-color: #252a32 !important;
        margin: 30px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666c76;
        font-size: 12px;
        margin-top: 45px;
    }

    /* Info box */
    .info-text {
        color: #777d87;
        font-size: 12px;
        line-height: 1.6;
        text-align: center;
        margin-top: 18px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = "models/signalscope_efficientnet_b0_v2_final.pth"


@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="logo">Signal<span class="logo-dot">Scope</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-generated image detection, powered by computer vision.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)


# --------------------------------------------------
# IMAGE + ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        use_container_width=True
    )

    st.markdown("<div style='height:12px'></div>",
                unsafe_allow_html=True)

    analyze = st.button(
        "Analyze image"
    )

    if analyze:

        with st.spinner("Analyzing..."):

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

            fake_probability = probabilities[0][0].item()
            real_probability = probabilities[0][1].item()

            if fake_probability > real_probability:

                prediction = "AI-GENERATED"
                confidence = fake_probability

            else:

                prediction = "REAL"
                confidence = real_probability


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Analysis result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{prediction}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence <strong>{confidence * 100:.2f}%</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        # REAL probability
        st.markdown(
            '<div class="prob-title">REAL</div>',
            unsafe_allow_html=True
        )

        st.progress(real_probability)

        # AI probability
        st.markdown(
            '<div class="prob-title">AI-GENERATED</div>',
            unsafe_allow_html=True
        )

        st.progress(fake_probability)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.markdown(
            '<div class="info-text">'
            'SignalScope provides a probabilistic estimate. '
            'It should not be treated as definitive proof of image origin.'
            '</div>',
            unsafe_allow_html=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    'SignalScope · Computer Vision Research Project'
    '</div>',
    unsafe_allow_html=True
)