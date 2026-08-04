"""
Módulo para treino e avaliação de modelos
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np

from src.utils.config import config

logger = logging.getLogger(__name__)


class Trainer:
    """
    Gerencia o treino e avaliação do modelo
    """
    
    def __init__(
        self,
        modelo: nn.Module,
        dispositivo: torch.device,
        config_instance=config
    ):
        self.modelo = modelo
        self.dispositivo = dispositivo
        self.config = config_instance
        
        # Configurar critério e otimizador
        self.criterio = nn.CrossEntropyLoss()
        self.otimizador = optim.Adam(
            self.modelo.fc.parameters(),
            lr=self.config.LEARNING_RATE
        )

        self.historico = {
            'perdas_treino': [],
            'acuracia_treino': [],
            'perdas_validacao': [],
            'acuracia_validacao': []
        }
    
    def treinar_epoch(
        self,
        loader_treino: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Treina o modelo por uma época
        
        Args:
            loader_treino: DataLoader de treino
            epoch: Número da época
        
        Retorna:
            Tuple[float, float]: (perda_média, acurácia)
        """
        self.modelo.train()
        perda_total = 0.0
        acertos_total = 0
        total_amostras = 0
        
        barra = tqdm(
            loader_treino,
            desc=f"Época {epoch}/{self.config.EPOCHS} [Treino]"
        )
        
        for imagens, rotulos in barra:
            imagens = imagens.to(self.dispositivo)
            rotulos = rotulos.to(self.dispositivo)
            
            imagens = self._redimensionar_imagens(imagens)
    
            self.otimizador.zero_grad()
            saidas = self.modelo(imagens)
            perda = self.criterio(saidas, rotulos)
            
 
            perda.backward()
            self.otimizador.step()
            

            perda_total += perda.item()
            _, preditos = torch.max(saidas, 1)
            acertos_total += (preditos == rotulos).sum().item()
            total_amostras += rotulos.size(0)
            
          
            barra.set_postfix({
                'Perda': f'{perda.item():.4f}',
                'Acurácia': f'{acertos_total/total_amostras:.4f}'
            })
        
        perda_media = perda_total / len(loader_treino)
        acuracia = acertos_total / total_amostras
        
        return perda_media, acuracia
    
    def validar(
        self,
        loader_validacao: DataLoader,
        descricao: str = "Validação"
    ) -> Tuple[float, float]:
        """
        Avalia o modelo nos dados de validação
        
        Args:
            loader_validacao: DataLoader de validação
            descricao: Descrição para o log
        
        Retorna:
            Tuple[float, float]: (perda_média, acurácia)
        """
        self.modelo.eval()
        perda_total = 0.0
        acertos_total = 0
        total_amostras = 0
        
        with torch.no_grad():
            for imagens, rotulos in tqdm(
                loader_validacao,
                desc=descricao,
                leave=False
            ):
                imagens = imagens.to(self.dispositivo)
                rotulos = rotulos.to(self.dispositivo)
                
                imagens = self._redimensionar_imagens(imagens)
                
                saidas = self.modelo(imagens)
                perda = self.criterio(saidas, rotulos)
                
                perda_total += perda.item()
                _, preditos = torch.max(saidas, 1)
                acertos_total += (preditos == rotulos).sum().item()
                total_amostras += rotulos.size(0)
        
        perda_media = perda_total / len(loader_validacao)
        acuracia = acertos_total / total_amostras
        
        return perda_media, acuracia
    
    def treinar(
        self,
        loader_treino: DataLoader,
        loader_validacao: DataLoader,
        epochs: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Executa o treino completo
        
        Args:
            loader_treino: DataLoader de treino
            loader_validacao: DataLoader de validação
            epochs: Número de épocas (opcional)
        
        Retorna:
            Dict: Histórico do treino
        """
        epochs = epochs or self.config.EPOCHS
        
        logger.info(f"Iniciando treino por {epochs} épocas...")
        logger.info(f"Dispositivo: {self.dispositivo}")
        
        for epoch in range(1, epochs + 1):
  
            perda_treino, acuracia_treino = self.treinar_epoch(
                loader_treino, epoch
            )
            

            perda_val, acuracia_val = self.validar(
                loader_validacao,
                f"Época {epoch} [Validação]"
            )
            

            self.historico['perdas_treino'].append(perda_treino)
            self.historico['acuracia_treino'].append(acuracia_treino)
            self.historico['perdas_validacao'].append(perda_val)
            self.historico['acuracia_validacao'].append(acuracia_val)

            logger.info(
                f"Época {epoch}: "
                f"Treino: {acuracia_treino:.4f} | "
                f"Val: {acuracia_val:.4f}"
            )
        
        logger.info("Treino concluído!")
        return self.historico
    
    def _redimensionar_imagens(self, imagens: torch.Tensor) -> torch.Tensor:
        """
        Redimensiona imagens para o tamanho esperado pelo modelo
        
        Args:
            imagens: Tensor com imagens [batch, canais, altura, largura]
        
        Retorna:
            torch.Tensor: Imagens redimensionadas
        """
        tamanho = self.config.TAMANHO_ENTRADA
        return torch.nn.functional.interpolate(
            imagens,
            size=(tamanho, tamanho),
            mode='bilinear',
            align_corners=False
        )


class ModelEvaluator:
    """
    Avalia o modelo com métricas detalhadas
    """
    
    def __init__(self, modelo: nn.Module, dispositivo: torch.device):
        self.modelo = modelo
        self.dispositivo = dispositivo
    
    def avaliar_detalhado(
        self,
        loader: DataLoader,
        classes: Tuple[str]
    ) -> Dict:
        """
        Avalia o modelo com métricas detalhadas
        
        Args:
            loader: DataLoader com dados
            classes: Nomes das classes
        
        Retorna:
            Dict: Métricas detalhadas
        """
        self.modelo.eval()
        
        # Matriz de confusão
        matriz_confusao = torch.zeros(
            len(classes), len(classes), dtype=torch.long
        )
        
        with torch.no_grad():
            for imagens, rotulos in tqdm(loader, desc="Avaliando"):
                imagens = imagens.to(self.dispositivo)
                rotulos = rotulos.to(self.dispositivo)
                
                # Redimensionar
                imagens = torch.nn.functional.interpolate(
                    imagens,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
                
                saidas = self.modelo(imagens)
                _, preditos = torch.max(saidas, 1)
                
                for verdadeiro, predito in zip(rotulos, preditos):
                    matriz_confusao[verdadeiro, predito] += 1
        
  
        metricas = {}
        for i, classe in enumerate(classes):
            verdadeiros_positivos = matriz_confusao[i, i].item()
            falsos_positivos = matriz_confusao[:, i].sum().item() - verdadeiros_positivos
            falsos_negativos = matriz_confusao[i, :].sum().item() - verdadeiros_positivos
            
            precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos + 1e-8)
            recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos + 1e-8)
            f1 = 2 * precisao * recall / (precisao + recall + 1e-8)
            
            metricas[classe] = {
                'precisao': precisao,
                'recall': recall,
                'f1': f1,
                'verdadeiros_positivos': verdadeiros_positivos
            }
        
     
        acuracia_geral = matriz_confusao.trace().item() / matriz_confusao.sum().item()
        
        return {
            'matriz_confusao': matriz_confusao.numpy(),
            'metricas_por_classe': metricas,
            'acuracia_geral': acuracia_geral
        }