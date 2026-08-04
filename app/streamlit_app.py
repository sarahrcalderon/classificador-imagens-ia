"""
Aplicação web com Streamlit
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import config
from src.utils.helpers import get_dispositivo
from src.models.model import ClassificadorImagens
from src.inference.predictor import ImagePredictor


# Configurar página
st.set_page_config(
    page_title="Classificador de Imagens",
    page_icon="🤖",
    layout="centered"
)

# Título
st.title("🤖 Classificador de Imagens")
st.markdown("""
    Envie uma imagem e o modelo vai tentar classificar entre **10 categorias**!
    
    Categorias: Avião, Automóvel, Pássaro, Gato, Veado, 
    Cachorro, Sapo, Cavalo, Navio, Caminhão
""")


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo com cache"""
    dispositivo = get_dispositivo()
    classificador = ClassificadorImagens()
    caminho_modelo = config.get_model_path('classificador.pth')
    
    try:
        modelo = classificador.carregar_modelo(str(caminho_modelo))
        return modelo, dispositivo, True
    except FileNotFoundError:
        return None, dispositivo, False
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, dispositivo, False


# Carregar modelo
modelo, dispositivo, modelo_carregado = carregar_modelo()
