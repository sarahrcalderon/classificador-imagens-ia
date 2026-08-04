"""
Módulo para definição e criação de modelos
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import logging

from src.utils.config import config

logger = logging.getLogger(__name__)


class ClassificadorImagens:
    """
    Wrapper para o modelo classificador de imagens
    """
    
    def __init__(self, num_classes: int = None, config_instance=config):
        self.config = config_instance
        self.num_classes = num_classes or self.config.NUM_CLASSES
        self._modelo = None
        self._dispositivo = None
    
    @property
    def dispositivo(self) -> torch.device:
        """Retorna o dispositivo (CPU/GPU)"""
        if self._dispositivo is None:
            self._dispositivo = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._dispositivo
    
    def criar_modelo(self) -> nn.Module:
        """
        Cria o modelo com Transfer Learning
        
        Retorna:
            nn.Module: Modelo pronto para treino
        """
        logger.info(f"Criando modelo {self.config.MODELO_BASE}...")

        modelo_base = self._carregar_modelo_base()
        
  
        self._congelar_pesos(modelo_base)
        

        self._substituir_camada_final(modelo_base)
        

        modelo_base = modelo_base.to(self.dispositivo)
        
        logger.info(f"Modelo criado no dispositivo: {self.dispositivo}")
        logger.info(f"Parâmetros treináveis: {self._contar_parametros_treinaveis(modelo_base):,}")
        
        self._modelo = modelo_base
        return modelo_base
    
    def _carregar_modelo_base(self) -> nn.Module:
        """Carrega o modelo base pré-treinado"""
        if self.config.MODELO_BASE == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Modelo {self.config.MODELO_BASE} não suportado")
    
    def _congelar_pesos(self, modelo: nn.Module) -> None:
        """Congela todos os pesos do modelo"""
        for parametro in modelo.parameters():
            parametro.requires_grad = False
        logger.info("Pesos do modelo base congelados")
    
    def _substituir_camada_final(self, modelo: nn.Module) -> None:
        """Substitui a camada fully connected final"""
        if hasattr(modelo, 'fc'):
            num_features = modelo.fc.in_features
            modelo.fc = nn.Linear(num_features, self.num_classes)
            logger.info(f"Camada final substituída: {num_features} -> {self.num_classes}")
        else:
            raise AttributeError("Modelo não possui camada 'fc'")
    
    def _contar_parametros_treinaveis(self, modelo: nn.Module) -> int:
        """Conta quantos parâmetros serão treinados"""
        return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    def carregar_modelo(self, caminho: str) -> nn.Module:
        """
        Carrega um modelo salvo do disco
        
        Args:
            caminho: Caminho para o arquivo .pth
        
        Retorna:
            nn.Module: Modelo carregado
        """
        if self._modelo is None:
            self.criar_modelo()
        
        logger.info(f"Carregando pesos de {caminho}...")
        self._modelo.load_state_dict(
            torch.load(caminho, map_location=self.dispositivo)
        )
        self._modelo.eval()
        logger.info("Modelo carregado com sucesso!")
        
        return self._modelo
    
    def salvar_modelo(self, caminho: str) -> None:
        """
        Salva o modelo no disco
        
        Args:
            caminho: Caminho para salvar o arquivo .pth
        """
        if self._modelo is None:
            raise ValueError("Modelo não foi criado ainda!")
        
        torch.save(self._modelo.state_dict(), caminho)
        logger.info(f"Modelo salvo em {caminho}")
    
    @property
    def modelo(self) -> nn.Module:
        """Retorna o modelo"""
        if self._modelo is None:
            self.criar_modelo()
        return self._modelo


def criar_classificador(num_classes: int = 10) -> ClassificadorImagens:
    """Cria uma instância do classificador"""
    return ClassificadorImagens(num_classes=num_classes)