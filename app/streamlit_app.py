"""
Aplicação web com Streamlit - Versão Compatível com Python 3.14+
"""

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import sys

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Classificador de Imagens",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Classificador de Imagens")

st.markdown("""
Envie uma imagem e o modelo vai classificar entre **10 categorias**:

✈️ Avião | 🚗 Automóvel | 🐦 Pássaro | 🐱 Gato | 🦌 Veado | 🐕 Cachorro | 🐸 Sapo | 🐴 Cavalo | 🚢 Navio | 🚛 Caminhão
""")

# ============================================
# CLASSES E MAPEAMENTO
# ============================================
CLASSES_CIFAR10 = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Veado', 
                   'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

# Mapeamento de classes do ImageNet para CIFAR-10
MAPEAMENTO_IMAGENET = {
    # Avião
    404: 'Avião', 405: 'Avião', 406: 'Avião', 407: 'Avião', 408: 'Avião',
    # Automóvel
    436: 'Automóvel', 437: 'Automóvel', 438: 'Automóvel', 439: 'Automóvel', 
    440: 'Automóvel', 441: 'Automóvel', 442: 'Automóvel', 443: 'Automóvel',
    # Pássaro
    8: 'Pássaro', 9: 'Pássaro', 10: 'Pássaro', 11: 'Pássaro', 12: 'Pássaro',
    13: 'Pássaro', 14: 'Pássaro', 15: 'Pássaro', 16: 'Pássaro', 17: 'Pássaro',
    18: 'Pássaro', 19: 'Pássaro', 20: 'Pássaro', 21: 'Pássaro', 22: 'Pássaro',
    23: 'Pássaro', 24: 'Pássaro', 80: 'Pássaro', 81: 'Pássaro', 82: 'Pássaro',
    83: 'Pássaro', 84: 'Pássaro', 85: 'Pássaro', 86: 'Pássaro', 87: 'Pássaro',
    88: 'Pássaro', 89: 'Pássaro', 90: 'Pássaro', 91: 'Pássaro', 92: 'Pássaro',
    93: 'Pássaro', 94: 'Pássaro', 95: 'Pássaro', 96: 'Pássaro', 97: 'Pássaro',
    98: 'Pássaro', 99: 'Pássaro', 100: 'Pássaro',
    # Gato
    281: 'Gato', 282: 'Gato', 283: 'Gato', 284: 'Gato', 285: 'Gato',
    286: 'Gato', 287: 'Gato', 288: 'Gato', 289: 'Gato', 290: 'Gato',
    291: 'Gato', 292: 'Gato', 293: 'Gato', 294: 'Gato', 295: 'Gato',
    # Veado
    341: 'Veado', 342: 'Veado', 343: 'Veado', 344: 'Veado', 345: 'Veado',
    346: 'Veado', 347: 'Veado', 348: 'Veado',
    # Cachorro
    151: 'Cachorro', 152: 'Cachorro', 153: 'Cachorro', 154: 'Cachorro',
    155: 'Cachorro', 156: 'Cachorro', 157: 'Cachorro', 158: 'Cachorro',
    159: 'Cachorro', 160: 'Cachorro', 161: 'Cachorro', 162: 'Cachorro',
    163: 'Cachorro', 164: 'Cachorro', 165: 'Cachorro', 166: 'Cachorro',
    167: 'Cachorro', 168: 'Cachorro', 169: 'Cachorro', 170: 'Cachorro',
    171: 'Cachorro', 172: 'Cachorro', 173: 'Cachorro', 174: 'Cachorro',
    175: 'Cachorro', 176: 'Cachorro', 177: 'Cachorro', 178: 'Cachorro',
    179: 'Cachorro', 180: 'Cachorro', 181: 'Cachorro', 182: 'Cachorro',
    183: 'Cachorro', 184: 'Cachorro', 185: 'Cachorro', 186: 'Cachorro',
    187: 'Cachorro', 188: 'Cachorro', 189: 'Cachorro', 190: 'Cachorro',
    191: 'Cachorro', 192: 'Cachorro', 193: 'Cachorro', 194: 'Cachorro',
    195: 'Cachorro', 196: 'Cachorro', 197: 'Cachorro', 198: 'Cachorro',
    199: 'Cachorro', 200: 'Cachorro', 201: 'Cachorro', 202: 'Cachorro',
    203: 'Cachorro', 204: 'Cachorro', 205: 'Cachorro', 206: 'Cachorro',
    207: 'Cachorro', 208: 'Cachorro', 209: 'Cachorro', 210: 'Cachorro',
    211: 'Cachorro', 212: 'Cachorro',
    # Sapo
    30: 'Sapo', 31: 'Sapo', 32: 'Sapo', 33: 'Sapo', 34: 'Sapo',
    349: 'Sapo', 350: 'Sapo',
    # Cavalo
    354: 'Cavalo', 355: 'Cavalo', 356: 'Cavalo', 357: 'Cavalo',
    # Navio
    779: 'Navio', 780: 'Navio', 781: 'Navio', 782: 'Navio', 783: 'Navio',
    784: 'Navio', 785: 'Navio', 786: 'Navio', 787: 'Navio',
    # Caminhão
    555: 'Caminhão', 556: 'Caminhão', 557: 'Caminhão', 558: 'Caminhão',
    559: 'Caminhão', 560: 'Caminhão', 561: 'Caminhão', 562: 'Caminhão',
    563: 'Caminhão', 564: 'Caminhão', 565: 'Caminhão', 566: 'Caminhão',
    567: 'Caminhão', 568: 'Caminhão', 569: 'Caminhão', 570: 'Caminhão',
    571: 'Caminhão', 572: 'Caminhão', 573: 'Caminhão', 574: 'Caminhão',
    575: 'Caminhão', 576: 'Caminhão', 577: 'Caminhão', 578: 'Caminhão',
    579: 'Caminhão', 580: 'Caminhão', 581: 'Caminhão', 582: 'Caminhão',
    583: 'Caminhão', 584: 'Caminhão', 585: 'Caminhão', 586: 'Caminhão',
    587: 'Caminhão', 588: 'Caminhão', 589: 'Caminhão', 590: 'Caminhão',
    591: 'Caminhão', 592: 'Caminhão',
}


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo ResNet18 pré-treinado"""
    try:
        # Verificar versão do Python
        st.info(f"🐍 Python {sys.version}")
        
        # Detectar dispositivo
        if torch.cuda.is_available():
            dispositivo = torch.device("cuda")
            st.success("🟢 GPU detectada!")
        else:
            dispositivo = torch.device("cpu")
            st.info("🟡 Usando CPU (mais lento)")
        
        # Carregar modelo - versão compatível
        modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modelo = modelo.to(dispositivo)
        modelo.eval()
        
        return modelo, dispositivo, True
    except Exception as e:
        st.error(f" Erro ao carregar modelo: {e}")
        return None, None, False

def traduzir_classe(idx):
    """Traduz o índice da classe ImageNet para CIFAR-10"""
    return MAPEAMENTO_IMAGENET.get(idx, None)

def preprocessar_imagem(imagem):
    """Preprocessa a imagem para o modelo"""
    transformacoes = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
    
    tensor = transformacoes(imagem)
    tensor = tensor.unsqueeze(0)
    return tensor


def classificar_imagem(modelo, tensor, dispositivo):
    """Classifica a imagem"""
    tensor = tensor.to(dispositivo)
    
    with torch.no_grad():
        saidas = modelo(tensor)
        probabilidades = torch.softmax(saidas, dim=1)
    
    # Top 5
    top5_prob, top5_idx = torch.topk(probabilidades, 5)
    top5_prob = top5_prob.cpu().numpy().flatten()
    top5_idx = top5_idx.cpu().numpy().flatten()
    
    # Tentar encontrar classe traduzida
    classe_encontrada = None
    prob_encontrada = 0
    
    for idx, prob in zip(top5_idx, top5_prob):
        classe = traduzir_classe(idx)
        if classe:
            return classe, prob * 100, top5_prob, top5_idx
    
    # Se não encontrou nenhuma classe mapeada
    return "Classe não mapeada", 0, top5_prob, top5_idx

def main():
    
    modelo, dispositivo, carregado = carregar_modelo()
    
    if not carregado:
        st.error(" Erro ao carregar o modelo. Tente novamente mais tarde.")
        st.stop()
    
    # Upload de imagem
    st.markdown("---")
    arquivo = st.file_uploader(
        " Selecione uma imagem",
        type=['jpg', 'jpeg', 'png']
    )
    
    # Imagens de exemplo
    st.markdown("###  Ou teste com uma imagem de exemplo:")
    
    exemplos = {
        " Gato": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
        " Cachorro": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/800px-Collage_of_Nine_Dogs.jpg",
        " Avião": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boeing_787-9_N29961_United_Airlines_%2848615635871%29.jpg/800px-Boeing_787-9_N29961_United_Airlines_%2848615635871%29.jpg",
        " Carro": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/2013_Volkswagen_Golf_VII_1.4.jpg/800px-2013_Volkswagen_Golf_VII_1.4.jpg",
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (nome, url) in enumerate(exemplos.items()):
        with [col1, col2, col3, col4][i]:
            if st.button(nome, key=f"exemplo_{i}", use_container_width=True):
                try:
                    response = requests.get(url, timeout=10)
                    imagem = Image.open(BytesIO(response.content))
                    
                    with st.spinner("Analisando..."):
                        tensor = preprocessar_imagem(imagem)
                        classe_prevista, confianca, top5_prob, top5_idx = classificar_imagem(
                            modelo, tensor, dispositivo
                        )
                    
                    st.image(imagem, caption=f"Exemplo: {nome}", width=200)
                    st.success(f" {classe_prevista}")
                    st.info(f"Confiança: {confianca:.1f}%")
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    # Processar imagem enviada
    if arquivo is not None:
        imagem = Image.open(arquivo)
        st.image(imagem, caption="📷 Sua imagem", width=300)
        
        if st.button(" Classificar!", type="primary", use_container_width=True):
            with st.spinner(" Analisando..."):
                try:
                    tensor = preprocessar_imagem(imagem)
                    classe_prevista, confianca, top5_prob, top5_idx = classificar_imagem(
                        modelo, tensor, dispositivo
                    )
                    
                    # Resultado
                    st.success(f" **Classe prevista: {classe_prevista}**")
                    st.info(f" **Confiança: {confianca:.1f}%**")
                    
                    # Top 5
                    st.markdown("### 📋 Top 5 classes")
                    
                    # Preparar dados para o gráfico
                    nomes = []
                    probs = []
                    
                    for i in range(5):
                        idx = top5_idx[i]
                        prob = top5_prob[i] * 100
                        nome = traduzir_classe(idx)
                        if nome:
                            nomes.append(nome)
                            probs.append(prob)
                        else:
                            nomes.append(f"Classe {idx}")
                            probs.append(prob)
                        
                        st.write(f"{i+1}. **{nomes[-1]}**: {probs[-1]:.1f}%")
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(8, 4))
                    cores = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]
                    ax.barh(nomes, probs, color=cores)
                    ax.set_xlabel('Probabilidade (%)')
                    ax.set_title('Top 5 Classes')
                    ax.set_xlim(0, 100)
                    
                    for i, prob in enumerate(probs):
                        ax.text(prob + 0.5, i, f'{prob:.1f}%', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()