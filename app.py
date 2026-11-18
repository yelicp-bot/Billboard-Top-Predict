import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Top Song Predictor & Player", page_icon="🎵", layout="centered")

# 1. Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_random_forest.joblib')

modelo = cargar_modelo()

st.title("🎵 Detector de Éxitos & Reproductor Musical")
st.write("¡Ingresa los detalles de tu canción favorita, conoce más sobre ella, escucha una muestra y predice si será un **Top 1**!")

# 2. Base de datos simulada de canciones para demostración interactiva
canciones_demo = {
    "Timber - Pitbull ft. Ke$ha": {
        "genre": "Urban",
        "mood": "Excited",
        "tempo": "Medium Tempo",
        "durationSeg": 223.0,
        "anioNacimiento": 1993.0,
        "artist_type": "Mixed",
        "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "img_url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=500&auto=format&fit=crop&q=60"
    },
    "Blank Space - Taylor Swift": {
        "genre": "Pop",
        "mood": "Energizing",
        "tempo": "Medium Tempo",
        "durationSeg": 231.0,
        "anioNacimiento": 1989.0,
        "artist_type": "Female",
        "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "img_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=60"
    },
    "Believer - Imagine Dragons": {
        "genre": "Rock",
        "mood": "Empowering",
        "tempo": "Medium Tempo",
        "durationSeg": 204.0,
        "anioNacimiento": 1987.0,
        "artist_type": "Mixed",
        "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "img_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=500&auto=format&fit=crop&q=60"
    }
}

# 3. Interfaz de usuario
st.header("🔍 Buscar o Introducir Canción")
opcion_cancion = st.selectbox("Selecciona una canción de ejemplo rápida o elige 'Personalizar'", ["Personalizar"] + list(canciones_demo.keys()))

# Variables por defecto
def_title = ""
def_artist = ""
def_genre = "Pop"
def_mood = "Energizing"
def_tempo = "Medium Tempo"
def_duration = 210
def_birth = 1990
def_artist_type = "Female"
def_audio = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3"
def_img = "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=60"

if opcion_cancion != "Personalizar":
    datos_demo = canciones_demo[opcion_cancion]
    partes = opcion_cancion.split(" - ")
    def_title = partes[0]
    def_artist = partes[1]
    def_genre = datos_demo["genre"]
    def_mood = datos_demo["mood"]
    def_tempo = datos_demo["tempo"]
    def_duration = int(datos_demo["durationSeg"])
    def_birth = int(datos_demo["anioNacimiento"])
    def_artist_type = datos_demo["artist_type"]
    def_audio = datos_demo["audio_url"]
    def_img = datos_demo["img_url"]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Información General")
    titulo = st.text_input("Título de la Canción", value=def_title if def_title else "Mi Canción Favorita")
    artista = st.text_input("Nombre del Artista", value=def_artist if def_artist else "Artista Estrella")
    genero = st.selectbox("Género Musical", ['Traditional', 'Pop', 'Urban', 'Alternative & Punk', 'Rock', 'Electronica', 'Jazz', 'Country', 'Other'], index=['Traditional', 'Pop', 'Urban', 'Alternative & Punk', 'Rock', 'Electronica', 'Jazz', 'Country', 'Other'].index(def_genre))
    mood = st.selectbox("Estado de ánimo (Mood)", ['Empowering', 'Cool', 'Energizing', 'Yearning', 'Defiant', 'Excited', 'Sensual', 'Urgent', 'Sophisticated', 'Rowdy', 'Brooding', 'Upbeat', 'Aggressive', 'Melancholy', 'Gritty', 'Romantic', 'Sentimental', 'Fiery', 'Easygoing', 'Peaceful', 'Stirring', 'Lively', 'Other'], index=['Empowering', 'Cool', 'Energizing', 'Yearning', 'Defiant', 'Excited', 'Sensual', 'Urgent', 'Sophisticated', 'Rowdy', 'Brooding', 'Upbeat', 'Aggressive', 'Melancholy', 'Gritty', 'Romantic', 'Sentimental', 'Fiery', 'Easygoing', 'Peaceful', 'Stirring', 'Lively', 'Other'].index(def_mood))

with col2:
    st.subheader("Atributos Técnicos")
    tempo = st.selectbox("Ritmo / Tempo", ['Slow Tempo', 'Medium Tempo', 'Fast Tempo'], index=['Slow Tempo', 'Medium Tempo', 'Fast Tempo'].index(def_tempo))
    duracion = st.slider("Duración (Segundos)", min_value=10, max_value=600, value=def_duration)
    anio_nacimiento = st.number_input("Año de Nacimiento del Artista", min_value=1930, max_value=2026, value=def_birth)
    tipo_artista = st.selectbox("Tipo de Artista", ['Male', 'Female', 'Mixed'], index=['Male', 'Female', 'Mixed'].index(def_artist_type))

# 4. Mostrar panel de la canción con reproducción multimedia e imagen
st.write("--- ")
st.subheader("🔊 Ficha de la Canción")

c_img, c_audio = st.columns([1, 2])
with c_img:
    st.image(def_img, caption=f"{titulo} - {artista}", use_container_width=True)
with c_audio:
    st.write(f"**Reproduciendo muestra de:** {titulo}")
    st.write(f"**Género:** {genero} | **Ánimo:** {mood}")
    st.audio(def_audio, format="audio/mp3")

# 5. Botón y lógica de Predicción
st.write("--- ")
if st.button("🔮 Analizar Posibilidades de Éxito"):
    # Preparar el DataFrame para coincidir con la codificación One-Hot del modelo
    registro = pd.DataFrame([{
        'chart_date': 20241026,
        'durationSeg': float(duracion),
        'anioNacimiento': float(anio_nacimiento),
        'mood': mood,
        'tempo': tempo,
        'genre': genero,
        'artist_type': tipo_artista
    }])

    # Columnas esperadas por el modelo entrenado
    columnas_entrenamiento = [
        'chart_date', 'durationSeg', 'anioNacimiento',
        'mood_Brooding', 'mood_Cool', 'mood_Defiant', 'mood_Easygoing',
        'mood_Empowering', 'mood_Energizing', 'mood_Excited', 'mood_Fiery',
        'mood_Gritty', 'mood_Lively', 'mood_Melancholy', 'mood_Other',
        'mood_Peaceful', 'mood_Romantic', 'mood_Rowdy', 'mood_Sensual',
        'mood_Sentimental', 'mood_Sophisticated', 'mood_Stirring',
        'mood_Upbeat', 'mood_Urgent', 'mood_Yearning',
        'tempo_Medium Tempo', 'tempo_Slow Tempo',
        'genre_Country', 'genre_Electronica', 'genre_Jazz', 'genre_Other',
        'genre_Pop', 'genre_Rock', 'genre_Traditional', 'genre_Urban',
        'artist_type_Male', 'artist_type_Mixed'
    ]

    registro_encoded = pd.get_dummies(registro, columns=['mood', 'tempo', 'genre', 'artist_type'])

    for col in columnas_entrenamiento:
        if col not in registro_encoded.columns:
            registro_encoded[col] = False

    registro_encoded = registro_encoded[columnas_entrenamiento]
    registro_encoded = registro_encoded.astype(int)

    # Realizar la predicción usando el modelo entrenado
    prediccion = modelo.predict(registro_encoded)[0]
    probabilidades = modelo.predict_proba(registro_encoded)[0]

    st.subheader("📈 Predicción del Algoritmo")
    if prediccion == 1:
        st.success(f"🎉 ¡Felicidades! El modelo predice que **'{titulo}'** será un **TOP HIT** con una probabilidad del {probabilidades[1]*100:.2f}%")
    else:
        st.warning(f"📉 El modelo predice que es probable que la canción no alcance los primeros puestos (Probabilidad del no-éxito: {probabilidades[0]*100:.2f}%)")
