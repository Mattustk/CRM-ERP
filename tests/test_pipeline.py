import pytest
import pandas as pd
import numpy as np
# Importa a função REAL do seu arquivo de produção
from src.pythonmain import validate_data_quality

@pytest.fixture
def data_mock():
    """Cria um DataFrame de teste com volumetria realista (normal) e injeção de anomalias"""
    np.random.seed(42)  # Idempotência computacional para testes unitários
    n_rows = 10000
    
    # Geração de volumetria normal limpa
    normal_data = {
        'id_transacao': [f'TRX-NORM-{i}' for i in range(n_rows)],
        'valor_unitario': np.random.uniform(10.0, 500.0, n_rows),
        'quantidade': np.random.randint(1, 10, n_rows),
        'custo_unitario': np.random.uniform(5.0, 250.0, n_rows),
    }
    df_normal = pd.DataFrame(normal_data)
    df_normal['valor_total_transacao'] = df_normal['valor_unitario'] * df_normal['quantidade']
    
    # Injeção controlada dos casos de borda originais
    error_data = pd.DataFrame({
        'id_transacao': ['TRX-001', 'TRX-002', 'TRX-003', 'TRX-004'],
        'valor_unitario': [100.0, -10.0, 50.0, 200.0],       # Erro no 2 (Negativo)
        'quantidade': [2, 1, 3, 2],
        'custo_unitario': [50.0, 5.0, None, 100.0],         # Erro no 3 (Custo Nulo)
        'valor_total_transacao': [200.0, -10.0, 150.0, 500.0] # Erro no 4 (Cálculo incorreto)
    })
    
    return pd.concat([df_normal, error_data], ignore_index=True)

def test_quality_gate_filters_negatives(data_mock):
    """Garante que valores negativos caiam na quarentena"""
    df_clean = validate_data_quality(data_mock, "TEST_NEGATIVOS")
    
    # O TRX-002 deve ter sido removido por ser negativo
    assert 'TRX-002' not in df_clean['id_transacao'].values

def test_quality_gate_filters_null_costs(data_mock):
    """Garante que custos nulos não passem para a camada Gold"""
    df_clean = validate_data_quality(data_mock, "TEST_NULOS")
    
    # O TRX-003 deve ter sido removido (Custo None)
    assert 'TRX-003' not in df_clean['id_transacao'].values
    assert df_clean['custo_unitario'].isnull().sum() == 0

def test_quality_gate_validates_math_integrity(data_mock):
    """Garante que erros de cálculo (Unit x Qtd != Total) sejam interceptados"""
    df_clean = validate_data_quality(data_mock, "TEST_MATEMATICA")
    
    # O TRX-004 (200 * 2 = 400, mas o total era 500) deve ser barrado
    assert 'TRX-004' not in df_clean['id_transacao'].values

def test_quality_gate_final_count(data_mock):
    """Garante que a volumetria limpa + TRX-001 sobrevivam intactos"""
    df_clean = validate_data_quality(data_mock, "TEST_FINAL")
    
    # Expectativa: 10.00
