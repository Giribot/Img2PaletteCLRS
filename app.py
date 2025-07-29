import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import json
from sklearn.cluster import KMeans
from datetime import datetime

def extract_colors(image, num_colors):
    image = image.convert("RGB")
    img_data = np.array(image).reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    kmeans.fit(img_data)
    return kmeans.cluster_centers_.astype(int)

def rgb_to_ipcolor(r, g, b):
    return int(f"0xFF{int(r):02X}{int(g):02X}{int(b):02X}", 16) - int("0x100000000", 16)

def generate_palette(image, name, num_colors):
    # Nom par défaut si vide
    if not name.strip():
        name = datetime.now().strftime("Palette_%Y%m%d_%H%M%S")

    colors = extract_colors(image, num_colors)
    values = [rgb_to_ipcolor(*c) for c in colors]
    data = {"colors": values, "name": name}
    filename = f"{name}.clrs"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    # Aperçu : grille avec max 15 colonnes
    cols = 15
    rows = (num_colors + cols - 1) // cols
    swatch_size = 60
    preview = Image.new("RGB", (cols * swatch_size, rows * swatch_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(preview)

    for i, (r, g, b) in enumerate(colors):
        x = (i % cols) * swatch_size
        y = (i // cols) * swatch_size
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=(int(r), int(g), int(b)))

    return f"Palette créée : {filename}", filename, preview

demo = gr.Interface(
    fn=generate_palette,
    inputs=[
        gr.Image(type="pil", label="Image (affichée en petit)", image_mode="RGB"),
        gr.Textbox(label="Nom de la palette (laisser vide = nom automatique)"),
        gr.Slider(2, 150, step=1, value=30, label="Nombre de couleurs")
    ],
    outputs=[
        gr.Text(label="Message"),
        gr.File(label="Fichier .clrs"),
        gr.Image(label="Aperçu de la palette extraite")
    ],
    title="Palette Infinite Painter",
    description="Génère un fichier .clrs depuis une image (jusqu’à 150 couleurs). Les images très grandes sont réduites pour éviter les lenteurs."
)

if __name__ == "__main__":
    demo.launch(prevent_thread_lock=True)
