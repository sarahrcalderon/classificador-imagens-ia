"""
Script para explorar o dataset
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import DataManager, get_class_names
from src.utils.helpers import set_seed


def main():
    """Função principal"""
    print("="*60)
    print(" EXPLORANDO O DATASET")
    print("="*60 + "\n")
    
    # Definir seed
    set_seed()
    
    # Carregar dados
    data_manager = DataManager()
    dataset_treino, dataset_teste = data_manager.carregar_datasets()
    classes = get_class_names()
    
    print(f"\n Dataset carregado com sucesso!")
    print(f" Imagens de treino: {len(dataset_treino)}")
    print(f" Imagens de teste: {len(dataset_teste)}")
    print(f" Classes: {', '.join(classes)}")
    
    # Mostrar distribuição das classes
    print("\n" + "="*60)
    print(" DISTRIBUIÇÃO DAS CLASSES")
    print("="*60)
    
    contagens = [0] * len(classes)
    for _, label in dataset_treino:
        contagens[label] += 1
    
    for classe, count in zip(classes, contagens):
        print(f"   {classe}: {count} imagens")
    
    # Mostrar exemplos
    print("\n" + "="*60)
    print(" AMOSTRAS DO DATASET")
    print("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(8):
        idx = np.random.randint(0, len(dataset_treino))
        imagem, label = dataset_treino[idx]
        
        # Desnormalizar
        imagem = imagem / 2 + 0.5
        imagem_np = imagem.numpy().transpose(1, 2, 0)
        
        axes[i].imshow(imagem_np)
        axes[i].set_title(classes[label], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('relatorios/amostras_dataset.png', dpi=150)
    print(" Amostras salvas em 'relatorios/amostras_dataset.png'")
    plt.show()
    
    print("\n" + "="*60)
    print(" EXPLORAÇÃO CONCLUÍDA!")
    print("="*60)


if __name__ == "__main__":
    main()