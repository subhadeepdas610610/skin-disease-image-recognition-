import json
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import models, transforms

CONDITION_INFO = {
    "nv": {
        "name": "Melanocytic Nevi (Common Mole)",
        "about": "A benign cluster of pigment-producing cells (melanocytes) on the skin, typically appearing as small, evenly colored spots or bumps.",
        "causes": "Driven by localized clustering of melanocytes; influenced by genetics and cumulative ultraviolet (UV) sunlight exposure.",
    },
    "mel": {
        "name": "Melanoma",
        "about": "A malignant skin tumor originating from melanocytes. Less common than benign lesions, but requires prompt clinical evaluation because it can metastasize if untreated.",
        "causes": "Associated with DNA damage in melanocytes—strongly linked to intense or intermittent UV exposure, fair skin, family history, or atypical moles.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "about": "Non-cancerous epidermal growths such as seborrheic keratoses, solar lentigines, or lichen planus-like keratoses.",
        "causes": "Commonly linked to skin aging and cumulative sun exposure rather than infection.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "about": "A common skin cancer arising from basal cells of the epidermis. Usually grows slowly and stays localized, though clinical treatment is required.",
        "causes": "Primarily associated with chronic or repeated exposure to ultraviolet (UV) radiation.",
    },
    "akiec": {
        "name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "about": "Rough, scaly patches or early intraepithelial lesions on sun-exposed skin that can represent precancerous change or early non-invasive carcinoma.",
        "causes": "Caused mainly by chronic cumulative UV exposure on areas such as the face, ears, forearms, or hands.",
    },
    "vasc": {
        "name": "Vascular Lesions",
        "about": "Benign vascular growths involving blood or lymph vessels (such as angiomas, angiokeratomas, or pyogenic granulomas), often reddish or purplish.",
        "causes": "Arise from localized proliferation or dilation of small blood or lymph vessels.",
    },
    "df": {
        "name": "Dermatofibroma",
        "about": "A small, firm, usually benign fibrous dermal nodule that often feels like a small hard bump beneath the skin.",
        "causes": "Frequently develops after minor localized skin trauma such as an insect bite or minor puncture.",
    },
}

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

with st.sidebar:
    st.header("Model & Evaluation Summary")
    st.markdown(
        """
- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Dataset:** HAM10000 (10,015 images; 7,470 unique `lesion_id`s)
- **Split:** 70% Train (7,017) / 15% Val (1,505) / 15% Test (1,493)
- **Preprocessing:** `Resize(256)` → `CenterCrop(224)` → ImageNet Normalize
- **Imbalance Handling:** Inverse-frequency weighted Cross-Entropy Loss
        """
    )
    st.divider()
    st.subheader("Held-Out Test Set Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Test Accuracy", "81.51%")
    m2.metric("Balanced Acc.", "73.36%")
    m3, m4 = st.columns(2)
    m3.metric("Macro-F1", "0.7028")
    m4.metric("Melanoma Recall", "59.49%")

st.title("Skin Lesion Classification System (HAM10000)")
st.caption("Trained on HAM10000 dermatoscopic images for research and reference; not a clinical diagnostic tool.")


@st.cache_resource
def load_model():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    state_dict = torch.load("skin_mobilenetv2.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


crop_preview_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Upload a dermatoscopic skin lesion image (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    cropped_preview = crop_preview_transform(image)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]

    probs_np = probs.cpu().numpy()
    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(class_names)))
    top_idx = int(top3_indices[0].item())
    top_prob = float(top3_probs[0].item())
    code = class_names[top_idx]
    confidence = top_prob * 100

    top3_dict = {
        class_names[int(idx.item())]: float(p.item())
        for p, idx in zip(top3_probs, top3_indices)
    }

    info = CONDITION_INFO.get(
        code,
        {"name": code, "about": "Predicted lesion category.", "causes": "Consult dermatological references."},
    )

    col1, col2 = st.columns([1, 1.4])
    with col1:
        img_tab1, img_tab2 = st.tabs(["Original Upload", "Model Input (224×224 Crop)"])
        with img_tab1:
            st.image(image, caption=f"Uploaded: {uploaded_file.name} ({image.size[0]}×{image.size[1]})", use_container_width=True)
        with img_tab2:
            st.image(cropped_preview, caption="After Resize(256) + CenterCrop(224)", use_container_width=True)

    with col2:
        st.subheader(f"Top Prediction: {code.upper()} — {info['name']}")
        st.metric("Top-1 Confidence", f"{confidence:.2f}%")
        st.markdown("**Clinical Overview:**")
        st.write(info["about"])
        st.markdown("**Associated Causes / Risk Factors:**")
        st.write(info["causes"])

        st.markdown("**Top-3 Class Probabilities:**")
        for c_code, c_prob in top3_dict.items():
            st.write(f"- `{c_code}` ({CONDITION_INFO.get(c_code, {}).get('name', c_code)}): **{c_prob * 100:.2f}%**")

    st.divider()
    st.subheader("Full 7-Class Probability Distribution")
    prob_df = pd.DataFrame({
        "Class": [f"{c.upper()} ({CONDITION_INFO.get(c, {}).get('name', c)})" for c in class_names],
        "Probability (%)": [round(float(p) * 100, 2) for p in probs_np],
    }).sort_values("Probability (%)", ascending=False)

    chart_col, table_col = st.columns([1.5, 1])
    with chart_col:
        st.bar_chart(prob_df.set_index("Class")["Probability (%)"])
    with table_col:
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
