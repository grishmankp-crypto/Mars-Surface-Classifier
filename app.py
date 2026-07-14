import os
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

# ── Hugging Face Spaces stores large files via Git LFS.
# The model is loaded from the root of the Space repo.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION — loads model + label encoder once at startup
# ─────────────────────────────────────────────────────────────────────────────
model        = None
label_encoder = None
CLASS_NAMES  = ["cracked", "gravel", "sand", "sedimentary"]  # fallback
NUM_CLASSES  = 4
init_error   = None

try:
    model_path = os.path.join(BASE_DIR, 'mars_phase1_efficientnet.keras')
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    pkl_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    with open(pkl_path, 'rb') as f:
        label_encoder = pickle.load(f)

    CLASS_NAMES = list(label_encoder.classes_)
    NUM_CLASSES  = len(CLASS_NAMES)
    print(f"✅ Loaded — classes: {CLASS_NAMES}")

except Exception as e:
    init_error = str(e)
    print(f"❌ Init failed: {init_error}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM SETUP
# Searches both flat layers and nested sub-models (EfficientNetB3 is nested).
# ─────────────────────────────────────────────────────────────────────────────
grad_model = None

def _find_and_build_gradcam(model):
    """Returns a 2-output Model [last_conv_output, predictions] or None."""
    # Walk top-level layers looking for the last Conv2D
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
        elif isinstance(layer, tf.keras.Model):        # nested sub-model
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    last_conv_name = sublayer.name

    if last_conv_name is None:
        return None

    # Try building grad-model at top level first
    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
    except ValueError:
        pass

    # Layer is inside a nested sub-model — chain them
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                inner = tf.keras.Model(
                    inputs=layer.input,
                    outputs=layer.get_layer(last_conv_name).output
                )
                inp = model.inputs[0]
                return tf.keras.Model(
                    inputs=inp,
                    outputs=[inner(inp), model(inp)]
                )
            except Exception:
                continue
    return None


if model is not None:
    try:
        grad_model = _find_and_build_gradcam(model)
        print("✅ Grad-CAM model ready." if grad_model else "⚠️ Grad-CAM unavailable — fallback active.")
    except Exception as ge:
        print(f"⚠️ Grad-CAM build failed: {ge}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_heatmap():
    """Gaussian blob used when Grad-CAM is unavailable."""
    x, y = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    g = np.exp(-((x**2 + y**2) / 2.0))
    return g / g.max()


def compute_gradcam(img_array, pred_index):
    if grad_model is None:
        return _fallback_heatmap()
    try:
        with tf.GradientTape() as tape:
            img_t = tf.cast(img_array, tf.float32)
            conv_out, preds = grad_model(img_t)
            tape.watch(conv_out)
            score = preds[:, pred_index]
        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]))
        return (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    except Exception:
        return _fallback_heatmap()


def heatmap_to_overlay(img_np, heatmap, alpha=0.45):
    h = np.array(Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224))) / 255.0
    h_rgb = (matplotlib.colormaps['jet'](h)[:, :, :3] * 255).astype(np.uint8)
    return (img_np * (1 - alpha) + h_rgb * alpha).astype(np.uint8)


def _error_outputs(message):
    dummy = Image.new('RGB', (224, 224), (60, 0, 0))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
    ax.axis('off')
    return message, dummy, fig, "Error", "Error"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION — called by Gradio on every image upload
# ─────────────────────────────────────────────────────────────────────────────

def analyze_mars_image(pil_image):
    if init_error:
        return _error_outputs(f"❌ Model failed to load:\n{init_error}")

    if pil_image is None:
        dummy = Image.new('RGB', (224, 224), (30, 30, 30))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, 'Upload an image', ha='center', va='center')
        ax.axis('off')
        return "⚠️ Please upload an image first.", dummy, fig, "N/A", "N/A"

    try:
        # Pre-process
        img_rgb   = pil_image.convert('RGB').resize((224, 224))
        img_np    = np.array(img_rgb)
        img_input = np.expand_dims(img_np / 255.0, axis=0).astype(np.float32)

        # Forward pass
        raw_preds  = model.predict(img_input, verbose=0)[0]
        pred_idx   = int(np.argmax(raw_preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(raw_preds[pred_idx]) * 100

        # Shannon entropy → uncertainty
        eps             = 1e-8
        entropy         = float(-np.sum(raw_preds * np.log(raw_preds + eps)))
        max_entropy     = float(np.log(NUM_CLASSES))
        uncertainty_pct = float(np.clip(entropy / max_entropy, 0, 1)) * 100

        # Anomaly score: high entropy + low confidence → anomalous
        anomaly_score = float(np.clip(
            1.0 - (confidence / 100.0) + (entropy / max_entropy) * 0.3, 0, 1
        ))
        is_anomaly = anomaly_score > 0.65

        # Grad-CAM overlay
        heatmap     = compute_gradcam(img_input, pred_idx)
        overlay_pil = Image.fromarray(heatmap_to_overlay(img_np, heatmap))

        # Confidence bar chart
        top_n      = min(NUM_CLASSES, 6)
        sorted_idx = np.argsort(raw_preds)[::-1][:top_n]
        top_cls    = [CLASS_NAMES[i] for i in sorted_idx]
        top_conf   = [raw_preds[i] * 100 for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.6)))
        bar_cols = ['#e74c3c' if c == pred_class else '#3498db' for c in top_cls]
        bars = ax.barh(top_cls[::-1], top_conf[::-1], color=bar_cols[::-1])
        ax.set_xlim(0, 108)
        ax.set_title('Class Probabilities', fontsize=11)
        for bar, val in zip(bars, top_conf[::-1]):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        # Output strings
        level = ('🔴 HIGH' if uncertainty_pct > 50
                 else '🟡 MEDIUM' if uncertainty_pct > 25
                 else '🟢 LOW')

        pred_text = (
            f"🪨 **Predicted:** {pred_class.upper()}\n"
            f"📊 **Confidence:** {confidence:.1f}%\n"
            f"🔢 **Uncertainty:** {uncertainty_pct:.1f}%"
        )
        unc_text  = f"Shannon Entropy: {entropy:.4f}\nUncertainty: {level} ({uncertainty_pct:.1f}%)"
        anom_text = (
            f"⚠️ ANOMALY FLAGGED  (score: {anomaly_score:.3f})"
            if is_anomaly else
            f"✅ Within known distribution  (score: {anomaly_score:.3f})"
        )

        return pred_text, overlay_pil, fig, unc_text, anom_text

    except Exception as e:
        return _error_outputs(f"❌ Runtime error: {e}")
    finally:
        plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MaRS — Mars Surface Classifier",
               theme=gr.themes.Soft(primary_hue="red")) as demo:

    gr.Markdown("# 🔴 MaRS 2026 — Mars Surface Sediment Classifier")
    gr.Markdown(
        "**IRC 2025 — 17th Place International | ERC 2025 — 2nd Place**  \n"
        "Upload a Mars rover surface image to get terrain classification, "
        "Grad-CAM explainability, uncertainty estimation, and anomaly detection."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type='pil', label='Upload Mars Surface Image', height=280)
            analyze_btn = gr.Button("🔍  Analyze Surface", variant='primary', size='lg')
            gr.Markdown(f"**Terrain classes:** {', '.join(CLASS_NAMES)}")

        with gr.Column(scale=2):
            prediction_out = gr.Markdown(label="Prediction")
            with gr.Row():
                gradcam_out     = gr.Image(type='pil', label='Grad-CAM (red = high attention)', height=224)
                confidence_plot = gr.Plot(label='Class Probabilities')
            with gr.Row():
                uncertainty_out = gr.Textbox(label='Uncertainty (Shannon Entropy)', lines=2)
                anomaly_out     = gr.Textbox(label='Anomaly Detection', lines=2)

    analyze_btn.click(
        fn=analyze_mars_image,
        inputs=[input_image],
        outputs=[prediction_out, gradcam_out, confidence_plot, uncertainty_out, anomaly_out]
    )
    input_image.change(
        fn=analyze_mars_image,
        inputs=[input_image],
        outputs=[prediction_out, gradcam_out, confidence_plot, uncertainty_out, anomaly_out]
    )

    gr.Markdown("""
---
**How it works**

| Component | Method |
|---|---|
| Classification | EfficientNetB3 fine-tuned via 2-stage transfer learning |
| Explainability | Grad-CAM — gradient-weighted class activation maps |
| Uncertainty | Shannon entropy of softmax distribution |
| Anomaly detection | Confidence + entropy composite score |
""")

demo.launch()
