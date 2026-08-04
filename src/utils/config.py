"""
Configurações centralizadas do projeto
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configurações do projeto"""
    
    # Diretórios
    RAIZ: Path = Path(__file__).parent.parent.parent
    DADOS_DIR: Path = RAIZ / "dados"
    MODELOS_DIR: Path = RAIZ / "models_salvos"
    RELATORIOS_DIR: Path = RAIZ / "relatorios"
    
    # Dataset
    NOME_DATASET: str = "CIFAR10"
    NUM_CLASSES: int = 10
    TAMANHO_IMAGEM: int = 32  # CIFAR-10 original
    
    # Modelo
    MODELO_BASE: str = "resnet18"
    TAMANHO_ENTRADA: int = 224  # ResNet espera 224x224
    
    # Treino
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 5
    NUM_WORKERS: int = 2
    SEED: int = 42
    

    CLASSES: tuple = ('Avião', 'Automóvel', 'Pássaro', 'Gato', 'Veado',
                     'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão')
    
    def __post_init__(self):
        """Cria diretórios se não existirem"""
        self.DADOS_DIR.mkdir(exist_ok=True)
        self.MODELOS_DIR.mkdir(exist_ok=True)
        self.RELATORIOS_DIR.mkdir(exist_ok=True)
    
    def get_model_path(self, nome: str = "classificador.pth") -> Path:
        """Retorna o caminho completo para salvar/carregar um modelo"""
        return self.MODELOS_DIR / nome
    
    def get_relatorio_path(self, nome: str) -> Path:
        """Retorna o caminho para salvar relatórios/gráficos"""
        return self.RELATORIOS_DIR / nome


config = Config()