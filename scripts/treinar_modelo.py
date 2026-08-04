"""
Script para treinar o modelo
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from src.utils.config import config
from src.utils.helpers import set_seed, plot_historico_treino, get_dispositivo
from src.data.dataset import DataManager
from src.models.model import ClassificadorImagens
from src.training.trainer import Trainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Função principal"""
    print("="*60)
    print(" TREINANDO O CLASSIFICADOR")
    print("="*60 + "\n")
    
    # 1. Configurar seed
    set_seed(config.SEED)
    
    # 2. Definir dispositivo
    dispositivo = get_dispositivo()
    logger.info(f"Usando dispositivo: {dispositivo}")
    
    # 3. Carregar dados
    logger.info("Carregando dados...")
    data_manager = DataManager()
    dataset_treino, dataset_teste = data_manager.carregar_datasets()
    loader_treino, loader_teste = data_manager.criar_dataloaders(
        dataset_treino, dataset_teste
    )
    
    # 4. Criar modelo
    logger.info("Criando modelo...")
    classificador = ClassificadorImagens()
    modelo = classificador.criar_modelo()
    
    # 5. Treinar
    logger.info("Iniciando treino...")
    trainer = Trainer(modelo, dispositivo)
    
    historico = trainer.treinar(
        loader_treino=loader_treino,
        loader_validacao=loader_teste
    )
    
    # 6. Salvar modelo
    caminho_modelo = config.get_model_path('classificador.pth')
    classificador.salvar_modelo(str(caminho_modelo))
    
    # 7. Visualizar resultados
    logger.info("Gerando gráficos...")
    plot_historico_treino(historico)
    
    # 8. Resumo final
    print("\n" + "="*60)
    print(" RESUMO DO TREINO")
    print("="*60)
    
    acuracia_final = historico['acuracia_validacao'][-1]
    print(f" Acurácia final: {acuracia_final*100:.2f}%")
    print(f" Modelo salvo em: {caminho_modelo}")
    
    print("\n" + "="*60)
    print(" TREINO CONCLUÍDO COM SUCESSO!")
    print("="*60)


if __name__ == "__main__":
    main()