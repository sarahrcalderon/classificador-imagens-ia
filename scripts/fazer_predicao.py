"""
Script para fazer uma predição com uma imagem
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
from src.utils.config import config
from src.utils.helpers import get_dispositivo
from src.models.model import ClassificadorImagens
from src.inference.predictor import ImagePredictor


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Classificar uma imagem')
    parser.add_argument('imagem', type=str, help='Caminho para a imagem')
    parser.add_argument('--top_k', type=int, default=3, help='Número de top classes')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔮 CLASSIFICANDO IMAGEM")
    print("="*60 + "\n")
    
    # 1. Definir dispositivo
    dispositivo = get_dispositivo()
    
    # 2. Carregar modelo
    print(f"\n Carregando modelo...")
    classificador = ClassificadorImagens()
    caminho_modelo = config.get_model_path('classificador.pth')
    
    try:
        modelo = classificador.carregar_modelo(str(caminho_modelo))
    except FileNotFoundError:
        print(f" Modelo não encontrado em {caminho_modelo}")
        print(" Execute 'scripts/treinar_modelo.py' primeiro")
        return
    
    # 3. Fazer predição
    print(f"\n Classificando: {args.imagem}")
    predictor = ImagePredictor(modelo, dispositivo)
    resultado = predictor.predizer(args.imagem, top_k=args.top_k)
    
    # 4. Mostrar resultados
    print("\n" + "="*60)
    print(" RESULTADO")
    print("="*60)
    
    print(f"\n Classe prevista: {resultado['classe_prevista']}")
    print(f" Confiança: {resultado['confianca']:.2f}%")
    
    print(f"\n Top {args.top_k} classes:")
    for i, (classe, prob) in enumerate(resultado['top_k'], 1):
        print(f"   {i}. {classe}: {prob:.2f}%")
    
    print("\n" + "="*60)
    print(" PREDIÇÃO CONCLUÍDA!")
    print("="*60)


if __name__ == "__main__":
    main()