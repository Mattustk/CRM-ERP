import pandas as pd
from faker import Faker
import random
import numpy as np
import os
import uuid

# --- REPRODUTIBILIDADE ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker('pt_BR')
Faker.seed(SEED)

# --- VARIÁVEIS DE AMBIENTE ---
FILE_TECH   = os.getenv('FILENAME_TECH',   'tech_nexus.csv')
FILE_RETAIL = os.getenv('FILENAME_RETAIL', 'retail_nexus.csv')

# --- CONFIGURAÇÕES ---
ids_vips          = [random.randint(100, 999) for _ in range(50)]
vendedores_tech   = [f"VEND-TECH-{i:02d}" for i in range(1, 11)]
vendedores_retail = [f"VEND-RETL-{i:02d}" for i in range(1, 21)]
metodos_pagamento = ['PIX', 'CARTAO_CREDITO', 'CARTAO_DEBITO', 'BOLETO']

catalogo = {
    'NEXUS TECH': {
        'Assinatura SaaS Standard':   {'valor': 1200.0,  'custo': 300.0,  'id': 'SOFT-SAAS-ST'},
        'Assinatura SaaS Premium':    {'valor': 2500.0,  'custo': 500.0,  'id': 'SOFT-SAAS-PR'},
        'Licença ERP Enterprise':     {'valor': 12500.0, 'custo': 3000.0, 'id': 'SOFT-ERP-ENT'},
        'Consultoria TI (Hora)':      {'valor': 250.0,   'custo': 100.0,  'id': 'SERV-CONS-H'},
        'Suporte Técnico Remoto':     {'valor': 450.0,   'custo': 150.0,  'id': 'SERV-SUP-REM'},
        'Projeto de Infraestrutura':  {'valor': 15000.0, 'custo': 7000.0, 'id': 'SERV-INFRA'},
        'Migração para Nuvem (AWS)':  {'valor': 8000.0,  'custo': 2500.0, 'id': 'SERV-CLOUD'},
        'Pentest de Segurança':       {'valor': 9500.0,  'custo': 4000.0, 'id': 'SEC-PENTEST'},
    },
    'NEXUS RETAIL': {
        'Cadeira Gamer Pro':      {'valor': 1850.0, 'custo': 900.0,  'id': 'HW-CHAIR-PR'},
        'Monitor 4K 27 pol':      {'valor': 2800.0, 'custo': 1400.0, 'id': 'HW-MONIT-4K'},
        'Teclado Mecânico RGB':   {'valor': 650.0,  'custo': 200.0,  'id': 'HW-KEYBD-RGB'},
        'Mouse Gamer Sem Fio':    {'valor': 450.0,  'custo': 180.0,  'id': 'HW-MOUSE-WL'},
        'Headset Surround 7.1':   {'valor': 890.0,  'custo': 350.0,  'id': 'HW-HSET-71'},
        'Notebook Ultra Slim':    {'valor': 5500.0, 'custo': 3200.0, 'id': 'HW-NOTE-ULT'},
        'PC Gamer Nexus 1.0':     {'valor': 7200.0, 'custo': 4500.0, 'id': 'HW-PC-GMR'},
        'PlayStation 5 Slim':     {'valor': 3800.0, 'custo': 2900.0, 'id': 'GME-PS5-SL'},
        'Cabo HDMI 2.1 2m':       {'valor': 120.0,  'custo': 30.0,   'id': 'ACC-HDMI-21'},
        'Mousepad Control XL':    {'valor': 150.0,  'custo': 45.0,   'id': 'ACC-MPAD-XL'},
    },
}


def _gerar_cpf_vetorizado(n: int) -> np.ndarray:
    """Gera n CPFs no formato XXX.XXX.XXX-XX sem validação de dígito — rápido."""
    d = np.random.randint(0, 10, size=(n, 11))
    return np.array([
        f"{r[0]}{r[1]}{r[2]}.{r[3]}{r[4]}{r[5]}.{r[6]}{r[7]}{r[8]}-{r[9]}{r[10]}"
        for r in d
    ])


def _gerar_emails_vetorizados(n: int) -> np.ndarray:
    """Gera n e-mails sintéticos sem chamar Faker por linha."""
    dominios = ['gmail.com', 'hotmail.com', 'yahoo.com.br', 'outlook.com', 'uol.com.br']
    sufixos  = np.random.randint(100, 9999, size=n)
    doms     = np.random.choice(dominios, size=n)
    prefixos = np.random.choice(
        ['user', 'cliente', 'comprador', 'nexus', 'vendas', 'contato'],
        size=n
    )
    return np.array([f"{p}{s}@{d}" for p, s, d in zip(prefixos, sufixos, doms)])


def _gerar_nomes_vetorizados(n: int) -> list:
    """Pré-gera um pool de nomes e reamostra — muito mais rápido que fake.name() por linha."""
    POOL = 5000
    pool = [fake.name() for _ in range(POOL)]
    return list(np.random.choice(pool, size=n))


def gerar_dados_escalado(
    nome_holding: str,
    num_registros: int,
    vendedores: list,
    arquivo_nome: str,
    chunk_size: int = 200_000,
) -> None:

    FIXED_END_DATE = '2025-12-31'
    produtos       = list(catalogo[nome_holding].keys())
    infos          = [catalogo[nome_holding][p] for p in produtos]

    if os.path.exists(arquivo_nome):
        os.remove(arquivo_nome)

    registros_gerados = 0
    header_necessario = True

    while registros_gerados < num_registros:
        n = min(chunk_size, num_registros - registros_gerados)

        # --- Campos vetorizados (numpy / list-comp em pool) ---
        idx_produto      = np.random.randint(0, len(produtos), size=n)
        item_nomes       = np.array(produtos)[idx_produto]
        valores_unit     = np.array([infos[i]['valor'] for i in idx_produto])
        custos_unit      = np.array([infos[i]['custo'] for i in idx_produto], dtype=float)
        ids_prod         = np.array([infos[i]['id']    for i in idx_produto])

        qtds             = np.random.randint(1, 6, size=n)
        probs            = np.random.random(size=n)

        # Ruído controlado — mesma lógica do original, vetorizado
        valor_total      = np.round(valores_unit * qtds, 2)
        valor_total      = np.where(probs < 0.02,  np.round(valor_total * 1.5, 2), valor_total)
        custos_unit      = np.where((probs >= 0.02) & (probs < 0.03), np.nan, custos_unit)
        mask_neg         = (probs >= 0.03) & (probs < 0.035)
        valores_unit     = np.where(mask_neg, -valores_unit, valores_unit)
        valor_total      = np.where(mask_neg, -valor_total,  valor_total)

        # Clientes: 60 % pool VIP, 40 % aleatório
        vip_mask         = np.random.random(size=n) > 0.4
        ids_cliente      = np.where(
            vip_mask,
            np.random.choice(ids_vips, size=n),
            np.random.randint(1000, 9000, size=n),
        )

        # Campos que ainda dependem de Faker — só timestamp (pool interno)
        timestamps = pd.date_range('2024-01-01', '2025-12-31', periods=n + 1)[:-1]
        timestamps = timestamps + pd.to_timedelta(np.random.randint(0, 86400, size=n), unit='s')
        datas      = timestamps.date

        df = pd.DataFrame({
            'holding':               nome_holding,
            'id_transacao':          [str(uuid.uuid4()) for _ in range(n)],
            'timestamp':             timestamps.strftime('%Y-%m-%d %H:%M:%S'),
            'id_cliente':            ids_cliente,
            'cpf_cliente':           _gerar_cpf_vetorizado(n),
            'nome_cliente':          _gerar_nomes_vetorizados(n),
            'email':                 _gerar_emails_vetorizados(n),
            'id_vendedor':           np.random.choice(vendedores, size=n),
            'id_produto':            ids_prod,
            'item_vendido':          item_nomes,
            'quantidade':            qtds,
            'valor_unitario':        valores_unit,
            'custo_unitario':        custos_unit,
            'valor_total_transacao': valor_total,
            'metodo_pagamento':      np.random.choice(metodos_pagamento, size=n),
            'data':                  datas,
        })

        df.to_csv(arquivo_nome, mode='a', index=False, header=header_necessario)
        header_necessario  = False
        registros_gerados += n

        pct = registros_gerados / num_registros * 100
        print(f"  [{nome_holding}] {registros_gerados:,} / {num_registros:,} ({pct:.1f}%)")

    print(f"\n✅ {nome_holding}: {num_registros:,} linhas → '{arquivo_nome}'\n")


# --- EXECUÇÃO ---
# Volumes aumentados — chunks maiores + geração vetorizada aguentam tranquilamente
VOLUME_TECH   = 3_000_000   # 3 M linhas
VOLUME_RETAIL = 5_000_000   # 5 M linhas

gerar_dados_escalado('NEXUS TECH',   VOLUME_TECH,   vendedores_tech,   FILE_TECH)
gerar_dados_escalado('NEXUS RETAIL', VOLUME_RETAIL, vendedores_retail, FILE_RETAIL)
