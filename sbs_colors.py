import streamlit as st
import math
from colormath.color_objects import sRGBColor, LabColor
import numpy as np
from skimage.color import rgb2lab
import json
import random
from PIL import Image
from lxml import etree

chosen_color = "#000000"

# Загрузка базы цветов из JSON
def load_colors_db(file_path="colors_db.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Сохранение базы цветов в JSON
def save_colors_db(colors, file_path="colors_db.json"):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(colors, file, indent=4, ensure_ascii=False)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

test_color = hex_to_rgb("#e1b9ca")

print(test_color, type(test_color))

def lab_distance(rgb1, rgb2):
    # Конвертируем RGB в LAB (используем scikit-image)
    lab1 = rgb2lab(np.array([[rgb1]], dtype=np.float32) / 255.0)
    lab2 = rgb2lab(np.array([[rgb2]], dtype=np.float32) / 255.0)
    
    # Вычисляем CIE76 (простое евклидово расстояние в LAB)
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

print(lab_distance(hex_to_rgb("#e1b9ca"), hex_to_rgb("#e3d6ab")))

def color_distance(rgb1, rgb2, method='euclidean'):
    if method == 'euclidean':
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))
    elif method == 'cie2000':
        return lab_distance(rgb1, rgb2)
    else:
        raise ValueError("Unknown method")

def find_closest_colors(target_hex, colors_db, num_results):
    target_rgb = hex_to_rgb(target_hex)
    closest = []
    
    for color in colors_db:
        color_rgb = color["rgb"]
        delta_e = lab_distance(target_rgb, color_rgb)
        closest.append((delta_e, color))
    
    closest.sort(key=lambda x: x[0])  # Сортируем по расстоянию
    return [color for (delta_e, color) in closest[:num_results]]

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

# -----------------------------
# Manual K-Means Implementation
# -----------------------------
def initialize_centroids(data, k):
    indices = random.sample(range(data.shape[0]), k)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    return np.array([
        data[labels == i].mean(axis=0) if np.any(labels == i)
        else data[random.randint(0, data.shape[0]-1)]
        for i in range(k)
    ])

def kmeans_from_scratch(data, k=5, max_iter=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return np.round(centroids).astype(int)

def get_items_by_color(thickness, sbs_code):
    """
    Ищет нитки Veritas по толщине и коду цвета SBS.
    
    :param thickness: Толщина нити (например, 50 в "50/2")
    :param sbs_code: Код цвета SBS (например, "101")
    :return: Список найденных предложений (ID, цвет, URL)
    """
    tree = etree.parse("threads.xml")  # Загружаем XML
    root = tree.getroot()
    
    matching_offers = []
    
    # Ищем offer, где vendor = Veritas, толщина = thickness/2, и цвет SBS = sbs_code
    xpath_query = f'''
        //offer[
            vendor="Veritas"
            and categoryId="1466"
            and .//param[@name="Толщина нити"]/text()="{thickness}/2"
            and .//param[@name="Цвет SBS (код)"]/text()="{sbs_code}"
        ]
    '''
    
    for offer in root.xpath(xpath_query):
        offer_id = offer.get('id')
        sbs_color = offer.xpath('.//param[@name="Цвет SBS (код)"]/text()')[0]
        url = offer.xpath('./url/text()')[0].replace("ru//", "ru/") if offer.xpath('./url/text()') else None
        
        matching_offers.append({
            "id": offer_id,
            "sbs_code": sbs_color,
            "url": url
        })
    
    return matching_offers

def apply_color(hex_code):
    global chosen_color 
    chosen_color = hex_code

with st.sidebar:
    uploaded_image = st.file_uploader("Загрузить образец", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Resize and flatten image for faster processing
        image = image.resize((200, 200))
        pixels = np.array(image).reshape(-1, 3)

        with st.spinner("Extracting colors..."):
            dominant_colors = kmeans_from_scratch(pixels, k=4)

        # -----------------------------
        # Display Result
        # -----------------------------
        st.markdown("## 🎨 Основные цвета")
        cols = st.columns(4)

        color_options = [rgb_to_hex(color) for color in dominant_colors]

        for i, color in enumerate(dominant_colors):
            hex_code = rgb_to_hex(color)
            rgb_clean = tuple(int(c) for c in color)
            
            with cols[i]:
                st.markdown(
                    f"<div style='background-color:{hex_code}; height:100px; border-radius:10px'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**HEX:** `{hex_code}`")
                st.markdown(f"**RGB:** `{rgb_clean}`")
                st.button(hex_code, on_click=apply_color(hex_code))

            st.success("Colors extracted successfully! 🎉")

        else:
            st.info("Please upload an image to start.")
  
colors_db = load_colors_db()  # Загружаем базу
target_color = st.color_picker("Выбор цвета", chosen_color)
num_results = st.number_input("Кол-во результатов", value=5)
closest_colors = find_closest_colors(target_color, colors_db, num_results)


cols = st.columns(num_results)

if target_color:
    for i, color in enumerate(closest_colors):
        st.markdown(
            f"<div style='background-color:{color['hex']}; height:100px; border-radius:10px'><div style='background-color: {target_color};'>Изначальный цвет</div></div>",
                unsafe_allow_html=True
            )
        f"https://welltex.ru/sbs/{color['name']}, `{color['hex']}`, `{color['name']}`"
            


