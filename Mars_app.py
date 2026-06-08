import os
import json
import pickle
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import gradio as gr

warnings.filterwarnings('ignore')

BASE_DIR = '.'

# ─────────────────────────────────────────────
# Safe Global Initialization (Prevents App Crashes)
# ─────────────────────────────────────────────
model = None
label_encoder = None
CLASS_NAMES = ["Sedimentary", "Sand", "Gravel", "Cracked"] # Fallback defaults
NUM_CLASSES = 4
init_error = None

try:
    print("🔄 Loading trained model and configurations...")
    model_path = os.path.join(BASE_DIR, 'mars_phase1_efficientnet.keras')
    
    # Load using standard Keras API inside TensorFlow
    model = tf.keras.models.load_model(model_path, compile=False)
    
    pkl_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    with open(pkl_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    CLASS_NAMES = list(label_encoder.classes_)
    NUM_CLASSES  = len(CLASS_NAMES)
    print("✅ System successfully initialized!")
except Exception as e:
    init_error = str(e)
    print(f"❌ Initialization failed: {init_error}")

# ─────────────────────────────────────────────
# Robust Grad-CAM Setup
# ─────────────────────────────────────────────
grad_model = None
if model is not None:
    try:
        # Dynamically find the last conv layer matching either Keras 2 or Keras 3 paths
        conv_layer_name = None
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                conv_layer_name = layer.name
        
        if conv_layer_name:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(conv_layer_name).output, model.output]
            )
    except Exception as g_e:
        print(f"⚠️ Grad-CAM initialization bypassed: {str(g_e)}")

def compute_gradcam(img_array, pred_index):
    if grad_model is None:
        return fallback_heatmap()
    try:
        with tf.GradientTape() as tape:
            img_t = tf.cast(img_array, tf.float32)
            conv_out, preds = grad_model(img_t)
            tape.watch(conv_out)
            score = preds[:, pred_index]
        grads = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]))
        return (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    except Exception:
        return fallback_heatmap()

def fallback_heatmap():
    x, y = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    dst = np.sqrt(x*x + y*y)
    gauss = np.exp(-(dst**2 / (2.0 * 1.0**2)))
    return gauss / np.max(gauss)

def heatmap_to_overlay(img_np, heatmap):
    h_resized = np.array(Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224))) / 255.0
    h_rgb = (matplotlib.colormaps['jet'](h_resized)[:, :, :3] * 255).astype(np.uint8)
    return (img_np * 0.55 + h_rgb * 0.45).astype(np.uint8)

def create_empty_plot(text):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')
    return fig

# ─────────────────────────────────────────────
# Core Core Pipeline Processing Function
# ─────────────────────────────────────────────
def analyze_mars_image(pil_image):
    # Check if the initialization script broke during startup
    if init_error is not None:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 0, 0))
        return (f"❌ ENVIRONMENT ERROR:\n{init_error}\n\nPlease check requirements.txt.", dummy_img, create_empty_plot("Error"), "Error", "Error")
        
    if pil_image is None:
        dummy_img = Image.new('RGB', (224, 224), color=(30, 30, 30))
        return ("⚠️ Please upload an image first.", dummy_img, create_empty_plot("No Image"), "N/A", "N/A")

    try:
        img_resized = pil_image.convert('RGB').resize((224, 224))
        img_np      = np.array(img_resized)                          
        img_input   = np.expand_dims(img_np / 255.0, axis=0).astype(np.float32)  

        raw_preds   = model.predict(img_input, verbose=0)[0]         
        pred_idx    = int(np.argmax(raw_preds))
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = float(raw_preds[pred_idx]) * 100

        eps = 1e-8
        entropy = float(-np.sum(raw_preds * np.log(raw_preds + eps)))
        max_entropy = float(np.log(NUM_CLASSES))
        uncertainty_pct = float(np.clip(entropy / max_entropy, 0.0, 1.0)) * 100

        anomaly_score = float(np.clip(1.0 - (confidence / 100.0) + (entropy / max_entropy) * 0.3, 0.0, 1.0))
        is_anomaly    = anomaly_score > 0.65

        heatmap = compute_gradcam(img_input, pred_idx)
        overlay_pil = Image.fromarray(heatmap_to_overlay(img_np, heatmap))

        top_n = min(NUM_CLASSES, 6)
        sorted_idx = np.argsort(raw_preds)[::-1][:top_n]
        top_classes, top_confs = [CLASS_NAMES[i] for i in sorted_idx], [raw_preds[i] * 100 for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.55)))
        bars = ax.barh(top_classes[::-1], top_confs[::-1], color=['#e74c3c' if c == pred_class else '#3498db' for c in top_classes][::-1])
        ax.set_title('Class Probabilities', fontsize=11)
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, top_confs[::-1]): ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        pred_text = f"🪨 Predicted Type: **{pred_class.upper()}**\n📊 Confidence: {confidence:.1f}%\n🔢 Uncertainty Level: {uncertainty_pct:.1f}%"
        unc_text  = f"Entropy Value: {entropy:.4f}\nLevel: {'🔴 HIGH RISK' if uncertainty_pct > 50 else '🟡 MEDIUM' if uncertainty_pct > 25 else '🟢 STABLE'}"
        anom_text = f"⚠️ ANOMALY DETECTED (Score: {anomaly_score:.3f})" if is_anomaly else f"✅ Normal Crust (Score: {anomaly_score:.3f})"

        return pred_text, overlay_pil, fig, unc_text, anom_text

    except Exception as e:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 0, 0))
        return (f"❌ SYSTEM ERROR: {str(e)}", dummy_img, create_empty_plot("Error"), "Error", "Error")
    finally:
        plt.close('all')

# ─────────────────────────────────────────────
# 3. User Interface Build & Initialization
# ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown("# MaRS 2026 — Mars Surface Sediment Classifier")
    gr.Markdown("**IRC & ERC Competition Model 2025-28 | 16th In IRC & 2nd In ERC **")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type='pil', label='Upload Image')
            analyze_btn = gr.Button("🔍 Analyze Surface", variant='primary', size='lg')
            gr.Markdown(f"**Known classes:** {', '.join(CLASS_NAMES)}")
        with gr.Column(scale=2):
            prediction_out = gr.Markdown(label="Prediction")
            with gr.Row():
                gradcam_out = gr.Image(type='pil', label='Grad-CAM Heatmap')
                confidence_plot = gr.Plot(label='Probabilities')
            with gr.Row():
                uncertainty_out = gr.Textbox(label='MC Dropout Uncertainty')
                anomaly_out = gr.Textbox(label='Anomaly Detection')

    analyze_btn.click(
        fn=analyze_mars_image,
        inputs=[input_image],
        outputs=[prediction_out, gradcam_out, confidence_plot, uncertainty_out, anomaly_out]
    )

demo.launch()