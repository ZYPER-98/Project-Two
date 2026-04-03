import os
import uuid
import base64
import pickle
import logging
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# App
app = Flask(__name__, template_folder="templates")
CORS(app, origins=os.getenv("CORS_ORIGINS", "*"))

# Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
IMG_SIZE      = (224, 224)
DEBUG_MODE    = os.getenv("FLASK_ENV", "development") != "production"

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cell class info
CLASS_INFO = {
    "im_Dyskeratotic": {
        "name":        "Dyskeratotic",
        "description": "Abnormal keratinization occurring within the epithelium, often associated with squamous cell abnormalities.",
        "risk":        "Moderate",
        "color":       "#FF6B6B",
    },
    "im_Koilocytotic": {
        "name":        "Koilocytotic",
        "description": "HPV-related cellular changes characterised by cytoplasmic vacuolisation (koilocytosis) around the nucleus.",
        "risk":        "High",
        "color":       "#FF4757",
    },
    "im_Metaplastic": {
        "name":        "Metaplastic",
        "description": "Transitional cells found at the squamo-columnar junction; normal variant but worth monitoring.",
        "risk":        "Low",
        "color":       "#FFA502",
    },
    "im_Parabasal": {
        "name":        "Parabasal",
        "description": "Immature basal-layer cells. Presence in large numbers may indicate atrophy or inflammation.",
        "risk":        "Low",
        "color":       "#2ED573",
    },
    "im_Superficial-Intermediate": {
        "name":        "Superficial",
        "description": "Normal mature squamous cells from the superficial / intermediate epithelial layer.",
        "risk":        "Very Low",
        "color":       "#1E90FF",
    },
}

# Globals
MODELS    = []
WEIGHTS   = []
CLASSES   = list(CLASS_INFO.keys())  # Initialize with all classes by default


def load_models():
    global MODELS, WEIGHTS, CLASSES

    if not os.path.exists("ensemble_config.pkl"):
        log.warning("ensemble_config.pkl not found")
        CLASSES   = list(CLASS_INFO.keys())
        WEIGHTS   = [1.0]
        return

    with open("ensemble_config.pkl", "rb") as f:
        config = pickle.load(f)

    model_paths = config.get("model_paths", [])
    weights     = config.get("weights", [])
    class_idx   = config.get("class_indices", {})

    if not model_paths or not weights or not class_idx:
        log.error("Invalid ensemble_config.pkl")
        CLASSES   = list(CLASS_INFO.keys())
        WEIGHTS   = [1.0]
        return

    for c in class_idx:
        if c not in CLASS_INFO:
            raise ValueError(f"CLASS_INFO missing key: '{c}'")

    CLASSES = list(class_idx.keys())

    try:
        from tensorflow.keras.models import load_model
        loaded = []
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: '{path}'")
            log.info("Loading model: %s", path)
            loaded.append(load_model(path))

        MODELS  = loaded
        WEIGHTS = weights
        log.info("Loaded %d model(s). Classes: %s", len(MODELS), CLASSES)
    except Exception as exc:
        log.error("Failed to load models: %s", exc)
        CLASSES   = list(CLASS_INFO.keys())
        WEIGHTS   = [1.0]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_ensemble(img_array):
    n     = len(CLASSES)
    probs = np.zeros(n, dtype=np.float32)

    # If no models loaded, generate random predictions for testing
    if not MODELS:
        raw   = np.abs(np.random.randn(n))
        probs = raw / raw.sum()
    else:
        for model, w in zip(MODELS, WEIGHTS):
            probs += model.predict(img_array, verbose=0)[0] * w
        total = probs.sum()
        if total > 0:
            probs /= total

    top_idx   = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    info      = CLASS_INFO[top_class]

    all_predictions = sorted(
        [
            {
                "class_display": CLASS_INFO[c]["name"],
                "confidence":    round(float(probs[i]) * 100, 2),
                "color":         CLASS_INFO[c]["color"],
            }
            for i, c in enumerate(CLASSES)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "predicted_class_display": info["name"],
        "confidence":              round(float(probs[top_idx]) * 100, 2),
        "description":             info["description"],
        "risk_level":              info["risk"],
        "color":                   info["color"],
        "all_predictions":         all_predictions,
    }


def image_to_base64(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html", class_info=CLASS_INFO)


@app.route("/health")
def health():
    return jsonify({
        "status":        "running",
        "models_loaded": len(MODELS),
        "classes":       CLASSES,
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT)).upper()}"}), 400

    ext      = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
        log.info("Saved: %s", filepath)

        img_array            = preprocess_image(filepath)
        result               = run_ensemble(img_array)
        result["image_data"] = image_to_base64(filepath)

        return jsonify(result), 200

    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            log.info("Deleted: %s", filepath)


@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.json

        # Create PDF in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)

        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("CervAI — Cell Classification Report", title_style))
        story.append(Spacer(1, 0.2*inch))

        # Report metadata
        meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
        story.append(Spacer(1, 0.3*inch))

        # Prediction results section
        results_title = ParagraphStyle('SectionTitle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#00d4ff'), spaceAfter=12)
        story.append(Paragraph("Analysis Results", results_title))

        # Prediction table
        pred_color = data.get('color', '#00d4ff')
        pred_data = [
            ['Parameter', 'Value'],
            ['Predicted Class', data.get('predicted_class_display', 'N/A')],
            ['Confidence', f"{data.get('confidence', 0):.2f}%"],
            ['Risk Level', data.get('risk_level', 'N/A')],
        ]

        pred_table = Table(pred_data, colWidths=[2*inch, 2*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c2535')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#111827')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#111827'), colors.HexColor('#0e1420')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1c2535')),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.3*inch))

        # Description section
        story.append(Paragraph("Cell Type Description", results_title))
        desc_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=10, alignment=TA_LEFT)
        story.append(Paragraph(data.get('description', 'N/A'), desc_style))
        story.append(Spacer(1, 0.3*inch))

        # All predictions
        story.append(Paragraph("All Classifications", results_title))
        all_pred_data = [['Cell Type', 'Confidence']]
        for pred in data.get('all_predictions', []):
            all_pred_data.append([pred.get('class_display', 'N/A'), f"{pred.get('confidence', 0):.2f}%"])

        all_pred_table = Table(all_pred_data, colWidths=[3*inch, 1.5*inch])
        all_pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c2535')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#111827')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#111827'), colors.HexColor('#0e1420')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1c2535')),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        story.append(all_pred_table)
        story.append(Spacer(1, 0.4*inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph(
            "<b>⚠️ Medical Disclaimer:</b> This report is for research &amp; educational purposes only. "
            "It is <b>not</b> a certified medical device and must not be used for clinical diagnosis. "
            "Always consult qualified healthcare professionals.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)

        return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True,
                        download_name=f'CervAI_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')

    except Exception as exc:
        log.exception("PDF generation failed: %s", exc)
        return jsonify({"error": f"Failed to generate report: {exc}"}), 500


if __name__ == "__main__":
    load_models()
    log.info("CervAI starting → http://localhost:8080  [debug=%s]", DEBUG_MODE)
    app.run(host="0.0.0.0", port=8080, debug=DEBUG_MODE)
