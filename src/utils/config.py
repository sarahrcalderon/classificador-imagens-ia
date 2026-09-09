import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:

    RAIZ: Path = Path(__file__).parent.parent.parent
    DADOS_DIR: Path = RAIZ / "dados"
    MODELOS_DIR: Path = RAIZ / "models_salvos"
    RELATORIOS_DIR: Path = RAIZ / "relatorios"
    

    NOME_DATASET: str = "CIFAR10"
    NUM_CLASSES: int = 10
    TAMANHO_IMAGEM: int = 32  

    MODELO_BASE: str = "resnet18"
    TAMANHO_ENTRADA: int = 224  

    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 5
    NUM_WORKERS: int = 2
    SEED: int = 42
    

    CLASSES: tuple = ('Avião', 'Automóvel', 'Pássaro', 'Gato', 'Veado',
                     'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão')
    
    def __post_init__(self):
        self.DADOS_DIR.mkdir(exist_ok=True)
        self.MODELOS_DIR.mkdir(exist_ok=True)
        self.RELATORIOS_DIR.mkdir(exist_ok=True)
    
    def get_model_path(self, nome: str = "classificador.pth") -> Path:
        return self.MODELOS_DIR / nome
    
    def get_relatorio_path(self, nome: str) -> Path:
        return self.RELATORIOS_DIR / nome


config = Config()