import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from pathlib import Path
import io
import os

st.set_page_config(page_title="Crack Detection — Demo", layout="centered")

st.title("Crack Detection — Project Memory & Demo")

# Show project memory (primary content)
mem_path = Path(__file__).parent / "memory.md"
if mem_path.exists():
    st.markdown(mem_path.read_text())
else:
    st.info("No `memory.md` found. You can add notes to `web/memory.md`.")

st.markdown("---")
st.header("Try the model")
st.markdown("Upload an image to test the model or use the heuristic fallback if no model is available.")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def list_model_files():
    if not MODEL_DIR.exists():
        return []
    return [p.name for p in MODEL_DIR.iterdir() if p.is_file()]


@st.cache_resource
def try_load_torch_model(path):
    try:
        import torch
        model = torch.load(path, map_location=torch.device('cpu'))
        model.eval()
        return ('torch', model)
    except Exception:
        return (None, None)


def heuristic_predict(img: Image.Image):
    # Very lightweight proxy: edge density
    gray = ImageOps.grayscale(img)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    # normalize to 0-1
    val = (arr > 30).mean()
    # map value to probability that image is cracked (higher edge density -> more likely cracked)
    prob = float(np.clip((val - 0.03) / 0.15, 0.0, 1.0))
    label = "Cracked" if prob > 0.5 else "Non-cracked"
    return label, prob


def predict_with_torch(model, img: Image.Image):
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        probs = torch.softmax(out.squeeze(), dim=0).cpu().numpy()
    # assume binary class [non-cracked, cracked] or [cracked, non-cracked]
    if probs.size == 1:
        prob_cracked = float(probs[0])
    else:
        # choose the index with higher probability as 'cracked'
        prob_cracked = float(probs.max())
    label = "Cracked" if prob_cracked > 0.5 else "Non-cracked"
    return label, prob_cracked


files = list_model_files()
sel = st.selectbox("Model file in `models/` (leave blank to use heuristic)", [""] + files)

loaded = None
loader_type = None
if sel:
    model_path = MODEL_DIR / sel
    loader_type, loaded = try_load_torch_model(str(model_path))
    if loader_type is None:
        st.warning("Could not load selected model (unsupported format or missing deps). Using heuristic.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)

    if loaded is not None and loader_type == 'torch':
        try:
            label, prob = predict_with_torch(loaded, img)
        except Exception as e:
            st.error(f"Error during PyTorch inference: {e}")
            label, prob = heuristic_predict(img)
    else:
        label, prob = heuristic_predict(img)

    st.markdown("---")
    st.subheader("Prediction")
    st.write(f"**Label:** {label}")
    st.write(f"**Confidence:** {prob:.2f}")

    if label == 'Cracked':
        st.success("Model indicates the image likely contains a crack.")
    else:
        st.info("Model indicates the image likely does not contain a visible crack.")

    st.markdown("---")
    st.markdown("If you have a trained model, place it in the `models/` directory and select it above. The app attempts to load PyTorch (`.pt`, `.pth`) models. If not available, a simple edge-density heuristic is used so you can test the UI locally.")

else:
    st.info("Upload an image to see predictions. Or place a model in `models/` and select it above.")

st.sidebar.markdown("---")
st.sidebar.markdown("App created for local testing. To deploy, push to a GitHub repo and use Streamlit Community Cloud or another hosting service.")
