#  Classificador de Imagens com Transfer Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://classificador-imagens-ia.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Sobre o Projeto

Este é um classificador de imagens que utiliza **Transfer Learning** com a arquitetura **ResNet18** pré-treinada no ImageNet. O modelo foi adaptado para classificar imagens em **10 categorias** do dataset CIFAR-10.

**Inteligência Artificial, Visão Computacional e Machine Learning**.

###  Categorias Classificadas

| Categoria | Emoji | Categoria | Emoji |
|-----------|-------|-----------|-------|
| Avião | ✈️ | Cachorro | 🐕 |
| Automóvel | 🚗 | Sapo | 🐸 |
| Pássaro | 🐦 | Cavalo | 🐴 |
| Gato | 🐱 | Navio | 🚢 |
| Veado | 🦌 | Caminhão | 🚛 |

---

## Demonstração

A aplicação está disponível online no **Streamlit Cloud**:

 **[Acesse o Classificador de Imagens](https://classificador-imagens-ia.streamlit.app/)**

### Funcionalidades

- **Upload de imagens** do seu computador
-  **Imagens de exemplo** para teste rápido
-  **Visualização das probabilidades** em gráfico
-  **Top 5 classes** com porcentagens de confiança
-  **Respostas rápidas** com modelo otimizado


---

##  Tecnologias Utilizadas

| Tecnologia | Descrição |
|------------|-----------|
| **Python** | Linguagem principal |
| **PyTorch** | Framework de Deep Learning |
| **ResNet18** | Arquitetura de rede neural (Transfer Learning) |
| **Streamlit** | Framework para aplicações web |
| **Matplotlib** | Visualização de dados e gráficos |
| **Pillow** | Processamento de imagens |
| **NumPy** | Operações matemáticas |

---

##  Como Funciona

### 1. Transfer Learning

O projeto utiliza a técnica de **Transfer Learning**, que consiste em:

1. Pegar um modelo já treinado (ResNet18) com milhões de imagens
2. "Congelar" as camadas que já sabem reconhecer características gerais
3. Substituir a camada final para classificar 10 categorias específicas
4. Treinar apenas a última camada com o dataset CIFAR-10

### 2. Pipeline de Classificação:
Imagem → Pré-processamento → ResNet18 → Softmax → Probabilidades → Classe Prevista


### 3. Mapeamento de Classes

O modelo foi originalmente treinado para 1000 classes do ImageNet. Para adaptá-lo ao CIFAR-10, criamos um **mapeamento** que traduz as classes do ImageNet para as 10 categorias desejadas.

---

##  Instalação Local

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes)

### Passos

1. **Clone o repositório**
   
2. **Crie um ambiente virtual**
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate

3. **Instale as dependências**
4. **Execute a aplicação** = streamlit run streamlit_app.py

