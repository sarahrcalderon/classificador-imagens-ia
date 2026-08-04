"""
Módulo para carregar e preparar datasets
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
import logging

from src.utils.config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Gerencia o carregamento e preparação dos dados"""
    
    def __init__(self, config_instance=config):
        self.config = config_instance
        self._transformacoes_treino = None
        self._transformacoes_teste = None
    
    @property
    def transformacoes_treino(self):
        """Transformações para dados de treino (com aumento)"""
        if self._transformacoes_treino is None:
            self._transformacoes_treino = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.config.TAMANHO_IMAGEM, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return self._transformacoes_treino
    
    @property
    def transformacoes_teste(self):
        """Transformações para dados de teste (sem aumento)"""
        if self._transformacoes_teste is None:
            self._transformacoes_teste = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return self._transformacoes_teste
    
    def carregar_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Carrega os datasets de treino e teste
        
        Retorna:
            Tuple[Dataset, Dataset]: (dataset_treino, dataset_teste)
        """
        logger.info("Carregando datasets CIFAR-10...")
        
        dataset_treino = torchvision.datasets.CIFAR10(
            root=str(self.config.DADOS_DIR),
            train=True,
            download=True,
            transform=self.transformacoes_treino
        )
        
        dataset_teste = torchvision.datasets.CIFAR10(
            root=str(self.config.DADOS_DIR),
            train=False,
            download=True,
            transform=self.transformacoes_teste
        )
        
        logger.info(f"Dataset treino: {len(dataset_treino)} imagens")
        logger.info(f"Dataset teste: {len(dataset_teste)} imagens")
        
        return dataset_treino, dataset_teste
    
    def criar_dataloaders(
        self,
        dataset_treino: Dataset,
        dataset_teste: Dataset,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Cria os DataLoaders para treino e teste
        
        Args:
            dataset_treino: Dataset de treino
            dataset_teste: Dataset de teste
            batch_size: Tamanho do batch (opcional)
        
        Retorna:
            Tuple[DataLoader, DataLoader]: (loader_treino, loader_teste)
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        
        loader_treino = DataLoader(
            dataset_treino,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        loader_teste = DataLoader(
            dataset_teste,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"DataLoaders criados com batch_size={batch_size}")
        return loader_treino, loader_teste
    
    def obter_classes(self) -> Tuple:
        """Retorna os nomes das classes"""
        return self.config.CLASSES


def get_data_manager() -> DataManager:
    """Retorna uma instância do DataManager"""
    return DataManager()


def get_class_names() -> Tuple:
    """Retorna os nomes das classes"""
    return config.CLASSES