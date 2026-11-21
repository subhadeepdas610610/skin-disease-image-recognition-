import json
import matplotlib.cm as cm
import numpy as np
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


def get_input_warnings(pil_image: Image.Image, confidence: float) -> list[str]:
    warnings = []
    width, height = pil_image.size
    aspect_ratio = width / max(height, 1)

    if width < 100 or height < 100:
        warnings.append(f"Small image resolution ({width}×{height}); cropped or low-res inputs may be unreliable.")

    if aspect_ratio > 1.6 or aspect_ratio < 0.625:
        warnings.append(f"Unusual aspect ratio ({aspect_ratio:.2f}:1) for a cropped skin lesion image (e.g., widescreen screenshot or banner).")

    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    skin_like_mask = (
        (r > 0.20)
        & (g > 0.08)
        & (b > 0.04)
        & (r > g)
        & (r > b)
        & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 0.04)
    )
    skin_ratio = float(np.mean(skin_like_mask))
    if skin_ratio < 0.25:
        warnings.append(f"Low skin-tone coverage ({skin_ratio * 100:.1f}% of pixels); this may not be a skin lesion photo.")

    if confidence < 55.0:
        warnings.append(f"Low top-1 confidence ({confidence:.2f}%); the model may be uncertain or seeing an out-of-domain image.")

    return warnings


def generate_gradcam_overlay(model, input_tensor, base_rgb_image, target_idx, alpha=0.45):
    activations = []
    gradients = []
    target_layer = model.features[-1]

    def forward_hook(_module, _inp, out):
        activations.append(out)

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        logits = model(input_tensor)
        score = logits[0, target_idx]
        score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    acts = activations[0][0].detach().cpu().numpy()      # [C, H, W]
    grads = gradients[0][0].detach().cpu().numpy()     # [C, H, W]
    weights = np.mean(grads, axis=(1, 2))              # [C]

    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_pil = Image.fromarray(np.uint8(cam * 255)).resize(base_rgb_image.size, Image.Resampling.BILINEAR)
    cam_norm = np.asarray(cam_pil, dtype=np.float32) / 255.0
    heatmap_rgb = (cm.jet(cam_norm)[:, :, :3] * 255).astype(np.float32)

    base_np = np.asarray(base_rgb_image.convert("RGB"), dtype=np.float32)
    overlay_np = np.clip((1.0 - alpha) * base_np + alpha * heatmap_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay_np)


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
    input_tensor = transform(image).unsqueeze(0).requires_grad_(True)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].detach()

    probs_np = probs.cpu().numpy()
    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(class_names)))
    top_idx = int(top3_indices[0].item())
    top_prob = float(top3_probs[0].item())
    code = class_names[top_idx]
    confidence = top_prob * 100

    input_warnings = get_input_warnings(image, confidence)
    if input_warnings:
        st.warning(
            "**Possible Non-Lesion / Low-Confidence Input:**\n"
            + "\n".join(f"- {msg}" for msg in input_warnings)
            + "\n\n*This model was trained only on HAM10000 dermatoscopic skin lesion images.*"
        )

    gradcam_image = generate_gradcam_overlay(model, input_tensor, cropped_preview, top_idx)

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
        img_tab1, img_tab2, img_tab3 = st.tabs([
            "Original Upload",
            "Model Input (224×224 Crop)",
            "Grad-CAM Heatmap",
        ])
        with img_tab1:
            st.image(image, caption=f"Uploaded: {uploaded_file.name} ({image.size[0]}×{image.size[1]})", use_container_width=True)
        with img_tab2:
            st.image(cropped_preview, caption="After Resize(256) + CenterCrop(224)", use_container_width=True)
        with img_tab3:
            st.image(gradcam_image, caption=f"Grad-CAM for {code.upper()} ({confidence:.2f}%)", use_container_width=True)

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
