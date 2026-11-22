import pandas as pd
from scipy import stats
import itertools
import re

# 1. Carregar os dados
df = pd.read_csv("results_final_v2.csv")

# Função para padronizar os IDs de todos os modelos
def normalize_sample_id(s):
    s = re.sub(r"^test_", "", str(s))
    s = re.sub(r"_chunk_\d+$", "", s)
    parts = s.split("_")
    if len(parts) == 6 and parts[-1].isdigit() and len(parts[-1]) <= 4:
        s = "_".join(parts[:-1])
    return s

df["base_sample_id"] = df["sample_id"].apply(normalize_sample_id)

# Métricas analisadas
metrics = ["lve", "fdd", "jitter_ratio", "mve", "mav_ratio", "hf_ratio", "tc_ratio"]
metrics = [m for m in metrics if m in df.columns]

# 2. Agregar métricas por sequência completa (ponderado por frames)
def weighted_avg(group):
    weights = group["num_frames"]
    total_w = weights.sum()
    res = {}
    for m in metrics:
        res[m] = (group[m] * weights).sum() / total_w
    return pd.Series(res)

df_agg = (
    df.groupby(["model_name", "base_sample_id"])[metrics + ["num_frames"]]
    .apply(weighted_avg, include_groups=False)
    .reset_index()
)

models = sorted(df_agg["model_name"].unique())

# Lista para armazenar pares não significantes
non_significant_results = []
all_results = []

# 3. Execução dos Testes Pareados
for m in metrics:
    for m1, m2 in itertools.combinations(models, 2):
        df_m1 = df_agg[df_agg["model_name"] == m1].set_index("base_sample_id")[m]
        df_m2 = df_agg[df_agg["model_name"] == m2].set_index("base_sample_id")[m]
        
        paired = pd.concat([df_m1, df_m2], axis=1, keys=[m1, m2]).dropna()
        n_pairs = len(paired)
        
        if n_pairs >= 2:
            diff = paired[m1].mean() - paired[m2].mean()
            t_stat, p_val = stats.ttest_rel(paired[m1], paired[m2])
            
            record = {
                "Métrica": m.upper(),
                "Modelo 1": m1,
                "Modelo 2": m2,
                "N_Pares": n_pairs,
                "Diff (M1 - M2)": round(diff, 4),
                "p-valor": p_val,
                "Significante": p_val < 0.05
            }
            all_results.append(record)
            
            if p_val >= 0.05:
                non_significant_results.append(record)

# 4. Exibição do Resumo dos Pares NÃO Significantes
print("\n" + "=" * 80)
print("📌 RESUMO: PARES ESTATISTICAMENTE NÃO SIGNIFICANTES (p >= 0.05)")
print("   (Não há evidência estatística de diferença de desempenho entre esses pares)")
print("=" * 80)

if non_significant_results:
    df_non_sig = pd.DataFrame(non_significant_results)
    
    # Formatação do p-valor para exibição limpa
    df_non_sig["p-valor"] = df_non_sig["p-valor"].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else f"{p:.4e}")
    
    # Remove a coluna booleana e exibe
    df_non_sig = df_non_sig.drop(columns=["Significante"])
    print(df_non_sig.to_string(index=False))
else:
    print("Nenhum par não significante encontrado. Todas as comparações foram estatisticamente significantes (p < 0.05).")

print("=" * 80)