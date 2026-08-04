"""
Funções auxiliares para o projeto
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging

from src.utils.config import config

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Define sementes para reprodutibilidade
    
    Args:
        seed: Semente aleatória
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed definida: {seed}")


def plot_historico_treino(historico: Dict[str, list]) -> None:
    """
    Plota o histórico de treino
    
    Args:
        historico: Dicionário com perdas e acurácias
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Perda
    ax1.plot(historico['perdas_treino'], label='Treino', marker='o')
    ax1.plot(historico['perdas_validacao'], label='Validação', marker='s')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')
    ax1.set_title('Evolução da Perda')
    ax1.legend()
    ax1.grid(True)
    
    # Acurácia
    ax2.plot(historico['acuracia_treino'], label='Treino', marker='o')
    ax2.plot(historico['acuracia_validacao'], label='Validação', marker='s')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.set_title('Evolução da Acurácia')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Salvar
    caminho = config.get_relatorio_path('historico_treino.png')
    plt.savefig(caminho, dpi=150)
    logger.info(f"Gráfico salvo em {caminho}")
    
    plt.show()


def plot_matriz_confusao(matriz: np.ndarray, classes: Tuple[str]) -> None:
    """
    Plota a matriz de confusão
    
    Args:
        matriz: Matriz de confusão
        classes: Nomes das classes
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matriz, cmap='Blues')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    # Adicionar valores
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matriz[i, j]),
                   ha='center', va='center',
                   color='white' if matriz[i, j] > matriz.max() / 2 else 'black')
    
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão')
    
    plt.colorbar(im)
    plt.tight_layout()
    
    # Salvar
    caminho = config.get_relatorio_path('matriz_confusao.png')
    plt.savefig(caminho, dpi=150)
    logger.info(f"Matriz de confusão salva em {caminho}")
    
    plt.show()


def get_dispositivo() -> torch.device:
    """Retorna o dispositivo disponível (CPU/GPU)"""
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {dispositivo}")
    return dispositivo


def contar_parametros(modelo: torch.nn.Module) -> Dict[str, int]:
    """
    Conta o número de parâmetros do modelo
    
    Args:
        modelo: Modelo PyTorch
    
    Retorna:
        Dict: Total e treináveis
    """
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'treinaveis': treinaveis,
        'congelados': total - treinaveis
    }