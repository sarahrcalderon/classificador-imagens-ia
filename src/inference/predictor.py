"""
Módulo para fazer predições com o modelo treinado
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List, Dict, Union
from pathlib import Path
import logging

from src.utils.config import config

logger = logging.getLogger(__name__)


class ImagePredictor:
    """
    Faz predições de imagem usando o modelo treinado
    """
    
    def __init__(self, modelo: torch.nn.Module, dispositivo: torch.device):
        self.modelo = modelo
        self.dispositivo = dispositivo
        self.modelo.eval()
        
        self.transformacoes = transforms.Compose([
            transforms.Resize((config.TAMANHO_ENTRADA, config.TAMANHO_ENTRADA)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.classes = config.CLASSES
    
    def preprocessar(self, imagem: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocessa a imagem para o modelo
        
        Args:
            imagem: Caminho para a imagem ou objeto PIL Image
        
        Retorna:
            torch.Tensor: Tensor pronto para o modelo
        """
        # Carregar imagem
        if isinstance(imagem, (str, Path)):
            imagem = Image.open(imagem)
        
        # Converter para RGB
        if imagem.mode != 'RGB':
            imagem = imagem.convert('RGB')
        
        tensor = self.transformacoes(imagem)
        
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.dispositivo)
    
    def predizer(
        self,
        imagem: Union[str, Path, Image.Image],
        top_k: int = 3
    ) -> Dict:
        """
        Faz a predição da imagem
        
        Args:
            imagem: Caminho ou imagem PIL
            top_k: Número de top classes para retornar
        
        Retorna:
            Dict: Resultados da predição
        """

        tensor = self.preprocessar(imagem)
        

        with torch.no_grad():
            saidas = self.modelo(tensor)
            probabilidades = torch.softmax(saidas, dim=1)
        
  
        top_prob, top_idx = torch.topk(probabilidades, top_k)
        top_prob = top_prob.cpu().numpy().flatten()
        top_idx = top_idx.cpu().numpy().flatten()
        

        classe_prevista = self.classes[top_idx[0]]
        confianca = top_prob[0] * 100
        
  
        top_classes = [
            (self.classes[idx], prob * 100)
            for idx, prob in zip(top_idx, top_prob)
        ]
        

        todas_prob = probabilidades.cpu().numpy().flatten()
        todas_classes = [
            (self.classes[i], todas_prob[i] * 100)
            for i in range(len(self.classes))
        ]
        todas_classes.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'classe_prevista': classe_prevista,
            'confianca': confianca,
            'top_k': top_classes,
            'todas_probabilidades': todas_classes
        }
    
    def predizer_lote(
        self,
        imagens: List[Union[str, Path, Image.Image]],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Faz predições para múltiplas imagens
        
        Args:
            imagens: Lista de caminhos ou imagens PIL
            top_k: Número de top classes para retornar
        
        Retorna:
            List[Dict]: Resultados para cada imagem
        """
        return [self.predizer(img, top_k) for img in imagens]


def criar_predictor(
    modelo: torch.nn.Module,
    dispositivo: torch.device
) -> ImagePredictor:
    """Cria uma instância do predictor"""
    return ImagePredictor(modelo, dispositivo)