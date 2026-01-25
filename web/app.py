
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2

import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from pathlib import Path
import io
import os
import base64
import re
import src
from src.resnet18_binary import ResNet18Binary
from src.cnn_crack_custom import CrackCNNCustom

sys.modules['models'] = src
sys.modules['models.cnn_crack_custom'] = src.cnn_crack_custom
sys.modules['models.resnet18_binary'] = src.resnet18_binary

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

st.set_page_config(page_title="Crack Detection", layout="centered")

st.title("Crack Detection Project")

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def replace_images_in_markdown(markdown_text, base_path):
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_match(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        full_path = base_path / rel_path
        
        if full_path.exists():
            mime_type = "image/png" if full_path.suffix.lower() == '.png' else "image/jpeg"
            img_b64 = img_to_bytes(full_path)
            return f'<img src="data:{mime_type};base64,{img_b64}" alt="{alt_text}" style="max-width: 100%;">'
        else:
            return match.group(0)

    return re.sub(pattern, replace_match, markdown_text)

# Show project memory (primary content)
mem_path = Path(__file__).parent / "memory.md"
if mem_path.exists():
    md_content = mem_path.read_text(encoding='utf-8')
    processed_md = replace_images_in_markdown(md_content, mem_path.parent)
    st.markdown(processed_md, unsafe_allow_html=True)
else:
    st.info("No `memory.md` found. You can add notes to `web/memory.md`.")

st.markdown("---")
st.header("Try the model")
st.markdown("Upload an image to test the model or use the heuristic fallback if no model is available.")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

def list_model_files():
    if not MODEL_DIR.exists():
        return []
    return [p.name for p in MODEL_DIR.iterdir() if p.is_file() and p.suffix == '.pth']

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

def heuristic_predict(img: Image.Image,
                        min_length=10,
                        min_elongation=2,
                        max_bbox_ratio=6,
                        min_area=10):
    """Classify image as Cracked or Non-cracked using connected component analysis on Canny edges"""
    if img is None:
        return None
    
    # Convert PIL Image to grayscale numpy array
    npimg = cv2.cvtColor(np.array(img.convert('L')), cv2.COLOR_GRAY2BGR)
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    canny = cv2.Canny(npimg, 50, 150)
    
    # Morphological operations to clean and connect
    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Connected components analysis
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(canny)
    
    # Extract crack-like features from components
    crack_objects = 0
    total_length = 0.0
    crack_mask = np.zeros_like(canny, dtype=np.uint8)
    
    for i in range(1, num_labels):  # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        if w == 0 or h == 0:
            continue
        
        elongation = max(w, h) / (min(w, h) + 1e-6)
        bbox_ratio = area / (w * h + 1e-6)
        length = max(w, h)
        
        # Detect crack-like components
        is_crack_like = (
            length > min_length and
            elongation > min_elongation and
            area > min_area and
            bbox_ratio < max_bbox_ratio
        )
        
        if is_crack_like:
            crack_objects += 1
            total_length += length
            crack_mask[labels_map == i] = 255
    
    # Classification decision
    if crack_objects >= 1 and total_length > 50:
        label = "Cracked"
    else:
        label = "Non-cracked"
    prob = min(1.0, total_length / 200.0)
    return label, prob, crack_mask

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

def explain_with_canny(crack_mask, original_img):
    """
    Visualiza las líneas detectadas por Canny superpuestas en la imagen original
    """
    # Redimensionar la máscara al tamaño original de la imagen
    mask_resized = cv2.resize(crack_mask, (original_img.width, original_img.height))
    
    # Convertir imagen original a numpy array
    rgb_img = np.array(original_img.convert("RGB"))
    
    # Normalizar para visualización
    rgb_img_normalized = np.float32(rgb_img) / 255
    
    # Crear una máscara coloreada (rojo para las líneas detectadas)
    colored_mask = np.zeros_like(rgb_img, dtype=np.float32)
    colored_mask[:, :, 0] = mask_resized / 255.0  # Canal rojo
    
    # Combinar con la imagen original
    alpha = 0.6
    visualization = rgb_img_normalized * (1 - alpha) + colored_mask * alpha
    
    # Convertir a uint8 para visualización
    visualization = (visualization * 255).astype(np.uint8)
    
    return visualization

# --- UI PRINCIPAL ---

files = list_model_files()
sel = st.selectbox("Model file in `models/`", ["Canny Edge detection"] + files)

loaded = None
loader_type = None

if sel not in ["Canny Edge detection"]:
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
        st.image(img, caption="Original Image", width='stretch')

    input_tensor = None
    crack_mask = None

    if loaded is not None and loader_type == 'torch':
        try:
            label, prob, input_tensor = predict_with_torch(loaded, img)
        except Exception as e:
            st.error(f"Error during PyTorch inference: {e}")
            label, prob, crack_mask = heuristic_predict(img)
    else:
        label, prob, crack_mask = heuristic_predict(img)

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

    # --- SECCIÓN GRAD-CAM / CANNY ---
    if input_tensor is not None and GRAD_CAM_AVAILABLE:
        st.markdown("---")
        st.subheader("Visual Explanation (Grad-CAM)")
        st.write("Heatmap highlights the regions influencing the model's decision.")
        
        # Generar visualización
        cam_image = explain_with_gradcam(loaded, input_tensor, img)
        
        if cam_image is not None:
            with col2:
                st.image(cam_image, caption="Model Attention (Heatmap)", width='stretch')
        else:
            with col2:
                st.info("Grad-CAM visualization not available for this architecture.")
    
    elif crack_mask is not None:
        st.markdown("---")
        st.subheader("Visual Explanation (Canny Edge Detection)")
        st.write("Red overlay shows the edges detected by Canny algorithm used to classify the image.")
        
        # Generar visualización de Canny
        canny_visualization = explain_with_canny(crack_mask, img)
        
        if canny_visualization is not None:
            with col2:
                st.image(canny_visualization, caption="Detected Crack Lines (Canny)", width='stretch')
    
    elif input_tensor is not None and not GRAD_CAM_AVAILABLE:
        st.info("Install `grad-cam` via pip to see visual explanations.")

    st.markdown("---")
    st.markdown("If you have a trained model, place it in the `models/` directory.")

else:
    st.info("Upload an image to see predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("App created for crack detection")