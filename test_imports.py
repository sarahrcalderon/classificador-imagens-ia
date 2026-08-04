# test_imports.py
print("Testando importações...")

try:
    from src.utils.config import config
    print("✅ config importado")
except Exception as e:
    print(f"❌ config: {e}")

try:
    from src.data.dataset import DataManager
    print("✅ DataManager importado")
except Exception as e:
    print(f"❌ DataManager: {e}")

try:
    from src.models.model import ClassificadorImagens
    print("✅ ClassificadorImagens importado")
except Exception as e:
    print(f"❌ ClassificadorImagens: {e}")

print("Teste concluído!")