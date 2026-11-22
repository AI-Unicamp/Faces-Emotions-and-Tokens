#!/usr/bin/env python3
"""
Plataforma de Avaliação de Avatar 3D
Teste de escolha forçada com 3 etapas e Videos de exemplos explicativos.
"""

from flask import Flask, render_template, jsonify, request, session, send_from_directory
import os
import json
import csv
import random
import time
import uuid
import re
from glob import glob

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "avatar-eval-secret-key")

# Por padrão, o modo teste é falso; será controlado pelo argumento CLI --teste [1]
app.config["TEST_MODE"] = False

# ---------------------------------------------------------------------------
# Configurações de Estímulos
# ---------------------------------------------------------------------------
SYLLABLES_A = ["ABA", "AKA", "ACHA", "ADA", "AFA", "AGA", "AJA", "APA", "ASSA", "ATA", "AVA", "AZA"]
SYLLABLES_I = ["IBI", "IKI", "ICHI", "IDI", "IFI", "IGUI", "IJI", "IPI", "ISSI", "ITI", "IVI", "IZI"]
SYLLABLES_U = ["UBU", "UKU", "UCHU", "UDU", "UFU", "UGU", "UJU", "UPU", "USSU", "UTU", "UVU", "UZU"]
ALL_SYLLABLES = SYLLABLES_A + SYLLABLES_I + SYLLABLES_U

# Subgrupo reduzido de opções específicas para a Etapa 2 [1]
SYLLABLES_STAGE2 = ["ABA", "ACHA", "AGA", "ASSA", "AVA"]

EMOTION_MAP_PT_TO_EN = {
    "Raiva": "angry",
    "Tristeza": "sad",
    "Alegria": "happy",
    "Nao sei": "unknown"
}

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ALL_SAMPLES = []

def get_noise_rank(noise_str):
    n_str = noise_str.lower()
    if "clean" in n_str or "limpo" in n_str:
        return 999  # Áudio totalmente limpo (mais fácil, joga por último)
        
    match = re.search(r'(-?\d+)', n_str)
    if match:
        return int(match.group(1))
    return 0


def get_noise_description(noise_str):
    n_str = noise_str.lower()
    if "clean" in n_str or "limpo" in n_str:
        return " (Sem Ruido)"
        
    match = re.search(r'(-?\d+)', n_str)
    if match:
        val = int(match.group(1))
        if val <= -30:
            return " (Puro Ruido)"
        elif val <= -10:
            return " (Ruido Medio)"
        elif val <= -1:
            return " (Ruido Baixo)"
    return ""


# ---------------------------------------------------------------------------
# Descoberta de Arquivos (Suporta pares de Videos síncronos na Etapa 3)
# ---------------------------------------------------------------------------
def discover_samples():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    samples = []
    extensions = ["*.mp4", "*.webm", "*.ogg", "*.mov"]
    media_files = []
    for ext in extensions:
        media_files.extend(glob(os.path.join(SAMPLES_DIR, ext)))

    # Agrupador auxiliar de pares para a Etapa 3 [1]
    stage3_pairs = {}

    for filepath in sorted(media_files):
        base = os.path.splitext(os.path.basename(filepath))[0]
        parts = base.split("_")
        
        # Ignora arquivos de exemplo e de aquecimento (Nao entram no teste pontuado)
        if base.startswith("exemplo_") or base.startswith("aquecimento_"):
            continue
            
        if len(parts) < 2:
            print(f"[WARN] Nome de arquivo '{base}' fora do padrão esperado. Ignorado.")
            continue
        
        prefix = parts[0].lower()
        if prefix == "stage1":
            if len(parts) < 3:
                continue
            syllable = parts[1].upper()
            noise = "_".join(parts[2:])
            samples.append({
                "id": base,
                "stage": "stage1",
                "syllable": syllable,
                "noise": noise,
                "emotion": "none",
                "filename": os.path.basename(filepath)
            })
        elif prefix == "stage2":
            if len(parts) < 4:
                continue
            syllable = parts[1].upper()
            emotion = parts[2].lower()
            noise = "_".join(parts[3:])
            samples.append({
                "id": base,
                "stage": "stage2",
                "syllable": syllable,
                "noise": noise,
                "emotion": emotion,
                "filename": os.path.basename(filepath)
            })
        elif prefix == "stage3":
            # Padrão esperado para Etapa 3: stage3_<ID_OU_EMOÇÃO>_<dynamic|constant> [1]
            match = re.match(r'(?i)stage3_(.+?)_(dynamic|constant)$', base)
            if match:
                key = match.group(1)
                v_type = match.group(2).lower()
                stage3_pairs.setdefault(key, {})[v_type] = os.path.basename(filepath)

    # Consolida os pares válidos da Etapa 3 [1]
    for key, files in stage3_pairs.items():
        if "dynamic" in files and "constant" in files:
            samples.append({
                "id": f"stage3_{key}",
                "stage": "stage3",
                "syllable": "none",
                "noise": "none",
                "emotion": key.split("_")[-1] if "_" in key else key, # Tenta inferir emoção do nome do arquivo
                "filename_dynamic": files["dynamic"],
                "filename_constant": files["constant"]
            })
        else:
            print(f"[WARN] Par incompleto detectado para a Etapa 3 com a chave '{key}'. Ignorado.")
            
    print(f"[OK] Descobertos {len(samples)} arquivos/pares de mídia válidos para teste em samples/")
    return samples


def build_experiment_order(samples):
    s1_items = [s for s in samples if s["stage"] == "stage1"]
    s2_items = [s for s in samples if s["stage"] == "stage2"]
    s3_items = [s for s in samples if s["stage"] == "stage3"]

    # Ordenação Stage 1: Agrupa por Ruido (noisiest -> clean) e randomiza as sílabas dentro de cada faixa
    s1_by_rank = {}
    for item in s1_items:
        rank = get_noise_rank(item["noise"])
        s1_by_rank.setdefault(rank, []).append(item)
    stage1_ordered = []
    for rank in sorted(s1_by_rank.keys()): 
        group = s1_by_rank[rank]
        random.shuffle(group) 
        stage1_ordered.extend(group)

    # Ordenação Stage 2: Mesmo processo (noisiest -> clean) com randomização interna por participante
    s2_by_rank = {}
    for item in s2_items:
        rank = get_noise_rank(item["noise"])
        s2_by_rank.setdefault(rank, []).append(item)
    stage2_ordered = []
    for rank in sorted(s2_by_rank.keys()):
        group = s2_by_rank[rank]
        random.shuffle(group) 
        stage2_ordered.extend(group)

    # Ordenação Stage 3: Totalmente aleatória [1]
    random.shuffle(s3_items)

    # -----------------------------------------------------------------------
    # REGRA DO MODO TESTE (CLI --teste) [1]
    # -----------------------------------------------------------------------
    if app.config["TEST_MODE"]:
        stage1_ordered = stage1_ordered[:2]
        stage2_ordered = stage2_ordered[:2]
        s3_items = s3_items[:2]
        print("[INFO] Modo teste ativo: as etapas foram limitadas a 2 tentativas para validação rápida.")

    ordered_list = stage1_ordered + stage2_ordered + s3_items
    
    global_indices = []
    for item in ordered_list:
        idx = ALL_SAMPLES.index(item)
        global_indices.append(idx)

    # Metadados estruturados
    stage_boundaries = []
    current_offset = 0
    
    if stage1_ordered:
        stage_boundaries.append({
            "stage_id": "stage1",
            "label": "Etapa 1: Teste de Inteligibilidade",
            "description": "",
            # ---- Configuração do wizard de entrada da Etapa 1 ----
            # Todo o texto visível ao participante mora aqui, para facilitar edição.
            "wizard": {
                "example_intro": {
                    "title": "Como funciona a tarefa",
                    "paragraphs": [
                        "Nesta etapa, um avatar 3D vai pronunciar uma sílaba curta. Sua tarefa é identificar qual sílaba foi dita.",
                        "Cada sílaba tem o formato vogal-consoante-vogal, sempre com a mesma vogal dos dois lados (por exemplo, A-B-A). O que muda, e o que você deve identificar, é a consoante do meio.",
                        "Antes da sílaba, o avatar diz \"I said...\" (em inglês, \"eu disse\"). Isso é apenas uma deixa: ignore essa parte e preste atenção na sílaba que vem logo depois.",
                        "As sílabas Nao são palavras de nenhuma língua, são apenas sons. Mesmo que alguma lembre uma palavra do português, trate-a só como um som."
                    ],
                    "button": "Ver exemplos"
                },
                "examples_title": "Exemplos",
                "examples_note": "Repare que a sílaba é a mesma nos dois Videos. O que muda é apenas o nível de Ruido de fundo.",
                "examples": [
                    {
                        "label": "Exemplo 1 (Sem Ruido):",
                        "video_url": "/samples/exemplo_stage1_ibi_limpo.mp4",
                        "expected_answer": "IBI"
                    },
                    {
                        "label": "Exemplo 2 (Com Ruido Máximo):",
                        "video_url": "/samples/exemplo_stage1_ibi_ruido.mp4",
                        "expected_answer": "IBI"
                    }
                ],
                "warmup_intro": {
                    "title": "Agora é a sua vez de praticar",
                    "paragraphs": [
                        "Você vai praticar com alguns Videos. Eles ficam cada vez mais difíceis: nos últimos, às vezes é quase impossível entender só pelo som, e isso é esperado.",
                        "Quando o som ficar difícil, observe a boca e o rosto do avatar. O movimento ajuda a identificar a consoante. Dê sempre o seu melhor palpite.",
                        "Cada Video só pode ser assistido uma vez, então fique atento. Estes Videos de prática Nao contam para o resultado."
                    ],
                    "button": "Começar a prática"
                },
                "warmup": {
                    "options": SYLLABLES_I,
                    "correct": "ICHI",
                    "items": [
                        {"level_label": "Sem Ruido", "video_url": "/samples/aquecimento_ichi_limpo.mp4", "allow_replay": False},
                        {"level_label": "Ruido Medio", "video_url": "/samples/aquecimento_ichi_medio.mp4", "allow_replay": False},
                        {"level_label": "Ruido pesado", "video_url": "/samples/aquecimento_ichi_pesado.mp4", "allow_replay": True}
                    ]
                },
                "transition": {
                    "title": "Agora começa o teste",
                    "paragraphs": [
                        "A prática acabou. O teste de verdade começa agora, e a partir daqui não há mais feedback. O teste começa com vídeo com ruído máximo, e termina com vídeos sem ruído.",
                        "Atenção: agora as sílabas usam a vogal A. Você escolherá entre estas DOZE opções:"
                    ],
                    "options": SYLLABLES_A,
                    "button": "Iniciar o teste"
                }
            },
            "start": current_offset,
            "count": len(stage1_ordered)
        })
        current_offset += len(stage1_ordered)

    if stage2_ordered:
        stage_boundaries.append({
            "stage_id": "stage2",
            "label": "Etapa 2: Perturbação Emocional",
            "description": "Nesta fase, o avatar falará sílabas expressando diferentes emoções (como raiva, tristeza ou alegria). Tente focar na voz e no movimento labial para identificar a sílaba correta, agora dentre CINCO opções.",
            "note": "Atenção: Assim como na etapa anterior, o avatar dirá a frase <strong>\"I said...\"</strong> logo antes da sólio alvo.",
            "examples": [
                {
                    "label": "Exemplo 1 (Pertubação Felicidade):",
                    "video_url": "/samples/exemplo_stage2_limpo_feliz.mp4",
                    "expected_answer": "AVA"
                },
                {
                    "label": "Exemplo 2  (Pertubação Tristeza):",
                    "video_url": "/samples/exemplo_stage2_ruido_triste.mp4",
                    "expected_answer": "ABA"
                }
            ],
            "start": current_offset,
            "count": len(stage2_ordered)
        })
        current_offset += len(stage2_ordered)

    if s3_items:
        stage_boundaries.append({
            "stage_id": "stage3",
            "label": "Etapa 3: Teste de Preferência Emocional",
            "description": "Nesta etapa final, você verá dois vídeos do avatar lado a lado, com o mesmo áudio. Os dois mostram a mesma fala, mas com expressões faciais diferentes. Sua tarefa é escolher em qual deles a expressão do rosto está mais sincronizada com a emoção expressa pela voz, ao longo de todo o vídeo.",
            "note": "Atenção: você pode tocar cada vídeo quantas vezes quiser, separadamente ou os dois ao mesmo tempo. O áudio é igual nos dois. A escolha só é liberada depois que você assistir aos dois vídeos pelo menos uma vez. Se realmente não conseguir decidir, escolha \"Nao sei\".",
            "examples": [
                {
                    "label": "Exemplo de comparação:",
                    "video_url_left": "/samples/exemplo_stage3_dynamic.mp4",
                    "video_url_right": "/samples/exemplo_stage3_constant.mp4",
                    "expected_answer": "Qual dos vídeos representa o rosto que se expressa em maior sincronia com o sentimento da voz?"
                    #"expected_answer": "Em um dos Videos a expressão do rosto acompanha as mudanças de emoção da fala; no outro, ela fica mais parada. Toque os dois e escolha o que combinar melhor com o áudio."
                }
            ],
            "start": current_offset,
            "count": len(s3_items)
        })

    return global_indices, stage_boundaries


def get_options_for_trial(sample):
    stage = sample["stage"]
    if stage == "stage2":
        return SYLLABLES_STAGE2
    elif stage == "stage1":
        syl = sample["syllable"].upper()
        if syl in SYLLABLES_A or syl.startswith("A"):
            return SYLLABLES_A
        elif syl in SYLLABLES_I or (syl.startswith("I") and syl.endswith("I")):
            return SYLLABLES_I
        elif syl in SYLLABLES_U or syl.startswith("U"):
            return SYLLABLES_U
        return SYLLABLES_A
    else:
        # Respostas para o teste de preferência da Etapa 3 [1]
        # "Nao sei" é uma abstenção, para quem Nao consegue decidir a preferência.
        return ["Video da Esquerda", "Video da Direita", "Nao sei"]


# ---------------------------------------------------------------------------
# Gravação de Resultados
# ---------------------------------------------------------------------------
def save_result(user_id, trial_data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    write_header = not os.path.isfile(csv_path)
    
    row = {
        "user_id": user_id,
        "timestamp": trial_data["timestamp"],
        "trial_index": trial_data["trial_index"],
        "sample_id": trial_data["sample_id"],
        "stage": trial_data["stage"],
        "syllable": trial_data["syllable"],
        "noise": trial_data["noise"],
        "emotion": trial_data["emotion"],
        "response": trial_data["response"],
        "correct": trial_data["correct"],
        "reaction_time_ms": trial_data["reaction_time_ms"],
    }
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        
    json_path = os.path.join(RESULTS_DIR, f"user_{user_id}.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"user_id": user_id, "trials": []}
    data["trials"].append(trial_data)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Rotas Flask
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/api/start", methods=["POST"])
def api_start():
    user_id = str(uuid.uuid4())[:8]
    order, stage_boundaries = build_experiment_order(ALL_SAMPLES)
    
    session["user_id"] = user_id
    session["order"] = order
    session["stage_boundaries"] = stage_boundaries
    session["current"] = 0
    session["started_at"] = time.time()
    
    return jsonify({
        "user_id": user_id,
        "total_trials": len(order),
        "stages": stage_boundaries,
    })


@app.route("/api/trial_data", methods=["GET"])
def api_trial_data():
    idx = session.get("current", 0)
    order = session.get("order", [])
    if idx >= len(order):
        return jsonify({"error": "Nao há mais tentativas disponíveis"}), 400
        
    sample = ALL_SAMPLES[order[idx]]
    options_raw = get_options_for_trial(sample)
    
    # Determina a legenda descritiva do nível de Ruido
    noise_desc = ""
    if sample["stage"] in ["stage1", "stage2"]:
        noise_desc = get_noise_description(sample["noise"])
        
    # Inicializa as respostas da API padrão
    video_url_left = ""
    video_url_right = ""
    video_url_single = ""

    if sample["stage"] == "stage3":
        # RANDOMIZAÇÃO DOS LADOS (Esquerda / Direita) NA ETAPA 3 [1]
        left_type = random.choice(["dynamic", "constant"])
        right_type = "constant" if left_type == "dynamic" else "dynamic"
        
        # Salva o mapeamento temporário de lados na sessão do usuário [1]
        session["stage3_left"] = left_type
        session["stage3_right"] = right_type
        
        left_file = sample["filename_dynamic"] if left_type == "dynamic" else sample["filename_constant"]
        right_file = sample["filename_constant"] if left_type == "dynamic" else sample["filename_dynamic"]
        
        video_url_left = f"/samples/{left_file}"
        video_url_right = f"/samples/{right_file}"
    else:
        video_url_single = f"/samples/{sample['filename']}"
    
    return jsonify({
        "trial_index": idx + 1,
        "total_trials": len(order),
        "stage": sample["stage"],
        "video_url": video_url_single,
        "video_url_left": video_url_left,
        "video_url_right": video_url_right,
        "noise_desc": noise_desc,
        "options": options_raw,
    })


@app.route("/api/practice_respond", methods=["POST"])
def api_practice_respond():
    # Loga respostas do aquecimento (Nao pontuadas). Marcadas com stage="warmup"
    # para serem filtradas/excluídas na análise. Nao mexe no índice da sessão.
    data = request.json or {}
    user_id = session.get("user_id", "unknown")

    correct_answer = data.get("correct_answer", "")
    trial_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_index": -1,
        "sample_id": "warmup_" + str(data.get("level", "")),
        "stage": "warmup",
        "syllable": correct_answer,
        "noise": data.get("level", ""),
        "emotion": "none",
        "response": data.get("response", ""),
        "correct": bool(data.get("is_correct", False)),
        "reaction_time_ms": data.get("reaction_time_ms", 0),
    }
    save_result(user_id, trial_log)
    return jsonify({"ok": True})


@app.route("/api/respond", methods=["POST"])
def api_respond():
    data = request.json
    response = data.get("response")
    reaction_time_ms = data.get("reaction_time_ms", 0)
    
    idx = session.get("current", 0)
    order = session.get("order", [])
    user_id = session.get("user_id", "unknown")
    
    if idx >= len(order):
        return jsonify({"error": "Nao há mais tentativas pendentes"}), 400
        
    sample = ALL_SAMPLES[order[idx]]
    
    correct = False
    response_save_label = response

    if sample["stage"] in ["stage1", "stage2"]:
        correct = (response.upper() == sample["syllable"].upper())
    else:
        # VALIDAÇÃO DO VENCEDOR NA ETAPA 3 [1]
        left_type = session.get("stage3_left", "dynamic")
        right_type = session.get("stage3_right", "constant")

        if response == "Nao sei":
            # Abstenção: Nao é preferência por dinâmico nem por constante.
            # Fica como correct=False e é identificável pelo prefixo "Nao sei" na resposta;
            # na análise, filtre essas linhas antes de calcular a preferência por dinâmico.
            correct = False
            response_save_label = f"Nao sei (Esquerda={left_type} | Direita={right_type})"
        else:
            chose_dynamic = False
            if response == "Video da Esquerda" and left_type == "dynamic":
                chose_dynamic = True
            elif response == "Video da Direita" and right_type == "dynamic":
                chose_dynamic = True

            correct = chose_dynamic # "correct" em stage3 significa que o usuário preferiu o modelo Dinâmico [1]
            response_save_label = f"{response} (Esquerda={left_type} | Direita={right_type})"

    trial_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_index": idx + 1,
        "sample_id": sample["id"],
        "stage": sample["stage"],
        "syllable": sample["syllable"],
        "noise": sample["noise"],
        "emotion": sample["emotion"],
        "response": response_save_label,
        "correct": correct,
        "reaction_time_ms": reaction_time_ms,
    }
    
    save_result(user_id, trial_log)
    
    session["current"] = idx + 1
    done = (idx + 1) >= len(order)
    
    return jsonify({
        "trial_index": idx + 1,
        "total_trials": len(order),
        "done": done,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avatar Facial Evaluation Platform")
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    parser.add_argument("--teste", action="store_true", help="Ativa o modo teste (reduz o número de tentativas por etapa para facilitar validações rápida)")
    args = parser.parse_args()

    # Define o estado global do modo teste a partir do CLI argument
    app.config["TEST_MODE"] = args.teste

    ALL_SAMPLES = discover_samples()
    # Geração automática de mockups caso a pasta esteja vazia
    if not ALL_SAMPLES:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        dummies = [
            # Exemplos e aquecimento (contexto I, held-out, fora do teste pontuado)
            "exemplo_stage1_ibi_limpo.mp4", "exemplo_stage1_ibi_ruido.mp4",
            "aquecimento_ichi_limpo.mp4", "aquecimento_ichi_medio.mp4", "aquecimento_ichi_pesado.mp4",
            "exemplo_stage2.mp4",
            "exemplo_stage3_dynamic.mp4", "exemplo_stage3_constant.mp4",
            # Teste pontuado: somente contexto A
            "stage1_ABA_noise_-30dB.mp4", "stage1_ABA_noise_-13dB.mp4",
            "stage1_ABA_noise_-6dB.mp4", "stage1_ABA_clean.mp4",
            "stage1_AKA_noise_-30dB.mp4", "stage1_AKA_clean.mp4",
            "stage2_ABA_happy_noise_-13dB.mp4", "stage2_ABA_sad_clean.mp4", "stage2_ABA_angry_noise_-6dB.mp4",
            "stage3_ABA_happy_dynamic.mp4", "stage3_ABA_happy_constant.mp4",
            "stage3_ASSA_sad_dynamic.mp4", "stage3_ASSA_sad_constant.mp4"
        ]
        for name in dummies:
            path = os.path.join(SAMPLES_DIR, name)
            with open(path, "wb") as f:
                f.write(b"")
        ALL_SAMPLES = discover_samples()

    print(f"\n[OK] Servidor rodando. Abra http://localhost:5000 no seu navegador.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)