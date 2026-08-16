"""
MiVision Streamlit app. Upload an image, run the trained YOLOv8n model, and
view detections drawn as bounding boxes with class labels and confidence.

Usage:
    streamlit run app.py
"""

import io
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import streamlit as st
from inference import DEFAULT_CONF, DEFAULT_MODEL_PATH, detect, load_model

# Fixed, distinct palette so the same class always gets the same color.
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


def class_colors(model) -> dict[str, str]:
    names = [model.names[i] for i in sorted(model.names)]
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def draw_detections(image: Image.Image, detections: list[dict], colors: dict[str, str]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        color = colors.get(det["class_name"], "#ff4040")
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1 + 2, y1 - 2), label, fill="white", font=font)
    return annotated


def build_pdf_report(
    annotated: Image.Image,
    detections: list[dict],
    source_name: str | None,
    conf: float,
) -> bytes:
    """Build a PDF detection report with the annotated image and a results table."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MiVision Detection Report", styles["Title"]),
        Paragraph(f"Image: {source_name or 'unknown'}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Paragraph(f"Confidence threshold: {conf:.2f}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]

    img_buf = io.BytesIO()
    annotated.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)
    max_width = 6.5 * inch
    scale = min(1.0, max_width / annotated.width)
    story.append(RLImage(img_buf, width=annotated.width * scale, height=annotated.height * scale))
    story.append(Spacer(1, 0.25 * inch))

    if detections:
        story.append(Paragraph(f"Detections ({len(detections)})", styles["Heading2"]))
        table_data = [["Class", "Confidence", "Bounding box (x1, y1, x2, y2)"]]
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            x1, y1, x2, y2 = det["bbox"]
            table_data.append([
                det["class_name"],
                f"{det['confidence']:.2%}",
                f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
            ])
        table = Table(table_data, hAlign="LEFT", colWidths=[1.5 * inch, 1.2 * inch, 2.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
    else:
        story.append(Paragraph(f"No aircraft detected above the {conf:.2f} confidence threshold.", styles["Normal"]))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Disclaimer: The model's confidence depends on the quality of the uploaded image. "
        "Try uploading a higher-quality image for better detection results and confidence.",
        styles["Italic"],
    ))

    doc.build(story)
    return buf.getvalue()


st.set_page_config(page_title="MiVision - Military Aircraft Detection", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap');
    html, body, button, input, textarea, label,
    div, span, p, h1, h2, h3, h4, h5, h6, li, [role="menuitem"] {
        font-family: 'Geist', sans-serif !important;
    }
    [data-testid="stIconMaterial"], [class*="material-symbols"], i.material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("MiVision")
    st.caption("YOLOv8n · 17-class military aircraft detector · V1")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                      help="Default 0.4 is the peak of the validation F1-confidence curve.")
    st.divider()
    with st.expander("Classes (17)"):
        st.write(", ".join([
            "F16", "F18", "C130", "F35", "F15", "F22", "B2", "B52", "AH64",
            "V22", "J20", "C17", "F14", "Su57", "Mig31", "CH47", "Tejas",
        ]))
    st.divider()
    st.caption("mAP50 0.608 · Precision 0.628 · Recall 0.544")
    st.caption("See [README](README.md) for full results.")
    st.divider()
    st.caption("Light/dark mode: use the menu (top right) > Settings > Theme.")

st.title("Military Aircraft Detection")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

sample_paths = sorted(Path("sample_images").glob("*"))
if sample_paths:
    st.caption("Or try a sample image")
    sample_cols = st.columns(len(sample_paths))
    for col, path in zip(sample_cols, sample_paths):
        with col:
            st.image(str(path), use_container_width=True)
            if st.button("Use this image", key=f"sample_{path.name}", use_container_width=True):
                st.session_state.selected_sample = str(path)
                st.session_state.pop("uploaded_name", None)

source_name = uploaded.name if uploaded is not None else st.session_state.get("selected_sample")

if uploaded is not None:
    image = Image.open(uploaded)
elif st.session_state.get("selected_sample"):
    image = Image.open(st.session_state["selected_sample"])
else:
    image = None

if image is not None:
    run = st.button("Run Detection", type="primary", use_container_width=False)
    new_image = st.session_state.get("source_name") != source_name

    if new_image or run or "detections" not in st.session_state:
        model = get_model()
        with st.spinner("Running inference..."):
            st.session_state.detections = detect(model, image, conf=conf)
            st.session_state.colors = class_colors(model)
            st.session_state.source_name = source_name

    detections = st.session_state.get("detections", [])
    colors = st.session_state.get("colors", {})
    annotated = draw_detections(image, detections, colors)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected")
        st.image(annotated, use_container_width=True)

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Detections", len(detections))
    m2.metric("Avg. confidence", f"{(sum(d['confidence'] for d in detections) / len(detections)):.0%}" if detections else "-")
    m3.metric("Threshold", f"{conf:.2f}")

    if detections:
        st.subheader("Detections")
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            c1, c2, c3 = st.columns([1, 3, 2])
            c1.color_picker("", colors.get(det["class_name"], "#ff4040"), key=id(det), disabled=True, label_visibility="collapsed")
            c2.write(f"**{det['class_name']}**")
            c3.progress(det["confidence"], text=f"{det['confidence']:.2%}")
    else:
        st.info(f"No aircraft detected above the {conf:.2f} confidence threshold. Try lowering it in the sidebar.")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        png_buf = io.BytesIO()
        annotated.save(png_buf, format="PNG")
        st.download_button("Download annotated image", png_buf.getvalue(),
                            file_name="mivision_detection.png", mime="image/png",
                            use_container_width=True)
    with dl_col2:
        pdf_bytes = build_pdf_report(annotated, detections, source_name, conf)
        st.download_button("Download PDF report", pdf_bytes,
                            file_name="mivision_detection_report.pdf", mime="application/pdf",
                            use_container_width=True)
else:
    st.info("Upload an image or pick a sample above to get started.")

st.divider()
st.caption(
    "⚠️ Disclaimer: The model's confidence depends on the quality of the "
    "uploaded image. Try uploading a higher-quality image for better "
    "detection results and confidence."
)
