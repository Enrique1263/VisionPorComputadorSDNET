import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from pathlib import Path
import io
import os

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

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
        import torchvision
        # weights_only=False es necesario para modelos guardados completos
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return ('torch', model)
    except Exception as e:
        st.error(f"Error detallado al cargar PyTorch: {e}")
        return (None, None)

def heuristic_predict(img: Image.Image):
    # Very lightweight proxy: edge density
    gray = ImageOps.grayscale(img)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    val = (arr > 30).mean()
    prob = float(np.clip((val - 0.03) / 0.15, 0.0, 1.0))
    label = "Cracked" if prob > 0.5 else "Non-cracked"
    return label, prob, None

def predict_with_torch(model, img: Image.Image):
    import torch
    from torchvision import transforms
    
    # --- PREPROCESAMIENTO ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Guardamos la imagen transformada para pasarla luego a GradCAM
    x_tensor = transform(img.convert('RGB')).unsqueeze(0)
    
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
            
        # Para modelo binario con BCEWithLogitsLoss, usar sigmoid
        prob_cracked = torch.sigmoid(out).item()

    # --- LÓGICA DE CLASIFICACIÓN ---
    # Modelo binario: >0.5 es Cracked, <=0.5 es Non-cracked
    label = "Cracked" if prob_cracked > 0.5 else "Non-cracked"
    
    return label, prob_cracked, x_tensor

def explain_with_gradcam(model, input_tensor, original_img):
    """
    Genera el mapa de calor Grad-CAM
    """
    if not GRAD_CAM_AVAILABLE:
        return None

    import torch
    
    # 1. Identificar la capa objetivo.
    # TODO: Mejorar para más arquitecturas
    try:
        target_layers = [model.features[-1]]
    except AttributeError:
        try:
            target_layers = [model.layer4[-1]]
        except AttributeError:
            st.warning("No se pudo identificar la capa convolucional automáticamente para GradCAM.")
            return None

    # 2. Inicializar GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Generar el mapa (Targets=None apunta a la clase con mayor probabilidad automáticamente)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # 4. Preparar imagen de fondo para superponer
    # GradCAM necesita la imagen normalizada entre 0 y 1 y del mismo tamaño que el tensor (224x224)
    rgb_img = original_img.convert("RGB").resize((224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    # 5. Crear superposición
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
    return visualization

# --- UI PRINCIPAL ---

files = list_model_files()
sel = st.selectbox("Model file in `models/` (leave blank to use heuristic)", [""] + files)

loaded = None
loader_type = None

if sel:
    model_path = MODEL_DIR / sel
    loader_type, loaded = try_load_torch_model(str(model_path))
    if loader_type is None:
        st.warning("Could not load selected model. Using heuristic.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    
    # Columnas para mostrar antes/después si hay GradCAM
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    input_tensor = None

    if loaded is not None and loader_type == 'torch':
        try:
            label, prob, input_tensor = predict_with_torch(loaded, img)
        except Exception as e:
            st.error(f"Error during PyTorch inference: {e}")
            label, prob, input_tensor = heuristic_predict(img)
    else:
        label, prob, input_tensor = heuristic_predict(img)

    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Métricas visuales
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Label", label)
    metric_col2.metric("Probability (Cracked)", f"{prob:.2%}")

    if label == 'Cracked':
        st.error("⚠️ Model detects a crack.")
    else:
        st.success("✅ No crack detected.")

    # --- SECCIÓN GRAD-CAM ---
    if input_tensor is not None and GRAD_CAM_AVAILABLE:
        st.markdown("---")
        st.subheader("Visual Explanation (Grad-CAM)")
        st.write("Heatmap highlights the regions influencing the model's decision.")
        
        # Generar visualización
        cam_image = explain_with_gradcam(loaded, input_tensor, img)
        
        if cam_image is not None:
            with col2:
                st.image(cam_image, caption="Model Attention (Heatmap)", use_container_width=True)
        else:
            with col2:
                st.info("Grad-CAM visualization not available for this architecture.")
    
    elif input_tensor is not None and not GRAD_CAM_AVAILABLE:
        st.info("Install `grad-cam` via pip to see visual explanations.")

    st.markdown("---")
    st.markdown("If you have a trained model, place it in the `models/` directory.")

else:
    st.info("Upload an image to see predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("App created for local testing.")