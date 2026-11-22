#!/usr/bin/env python3
import os
import re
import shutil
from glob import glob

# --- CONFIGURAÇÃO ---
# Pasta onde seus arquivos originais estão atualmente localizados
INPUT_DIR = "videos_originais" 

# Pasta de destino estruturada (deve ser a mesma definida no seu app.py)
OUTPUT_DIR = "samples"

# Mantenha True para apenas testar e visualizar as mudanças no terminal.
# Mude para False quando estiver pronto para renomear e mover os arquivos reais.
DRY_RUN = False
# --------------------

SYLLABLES = [
    "ABA", "AKA", "ACHA", "ADA", "AFA", "AGA", "AJA", "APA", "ASSA", "ATA", "AVA", "AZA",
    "IBI", "IKI", "ICHI", "IDI", "IFI", "IGI", "IJI", "IPI", "ISSI", "ITI", "IVI", "IZI",
    "UBU", "UKU", "UCHU", "UDU", "UFU", "UGU", "UJU", "UPU", "USSU", "UTU", "UVU", "UZU"
]

EMOTIONS = ["happy", "sad", "angry", "alegria", "tristeza", "raiva"]

EMOTION_NORM_MAP = {
    "alegria": "happy",
    "tristeza": "sad",
    "raiva": "angry",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry"
}

def parse_and_rename(filename):
    base, ext = os.path.splitext(filename)
    
    # Remove o sufixo redundante "_video" se existir
    cleaned_base = re.sub(r'_video$', '', base, flags=re.IGNORECASE)
    cleaned_base = re.sub(r'_video_', '_', cleaned_base, flags=re.IGNORECASE)
    
    # -----------------------------------------------------------------------
    # REGRA PRIORITÁRIA PARA ETAPA 3 (Detecção de dynamic / constant) [1]
    # Se o arquivo contiver "dynamic" ou "constant", ele pertence à Etapa 3 [1].
    # Exemplo: id01_p01_e02_f29_constant_100_680.mp4 -> stage3_id01_p01_e02_f29_constant.mp4
    # -----------------------------------------------------------------------
    match_s3 = re.search(r"(?i)(.+?)_(dynamic|constant)", cleaned_base)
    if match_s3:
        s3_id = match_s3.group(1)            # Extrai o identificador (ex: id01_p01_e02_f29) [1]
        s3_type = match_s3.group(2).lower()  # Extrai se é dynamic ou constant [1]
        new_name = f"stage3_{s3_id}_{s3_type}{ext}"
        return "Etapa 3", new_name

    # 1. Identificar Sílaba (Para as etapas 1 e 2)
    found_syllable = None
    for syl in SYLLABLES:
        # Busca a sílaba exata separada por underlines
        if f"_{syl}_" in f"_{cleaned_base}_".upper():
            found_syllable = syl
            break

    # 2. Identificar Emoção
    found_emotion = None
    for emo in EMOTIONS:
        if emo in cleaned_base.lower():
            found_emotion = emo
            break
            
    norm_emotion = EMOTION_NORM_MAP.get(found_emotion) if found_emotion else None

    # 3. Limpar string para extrair nível de ruído
    noise_part = cleaned_base
    if found_syllable:
        noise_part = re.sub(rf"(?i){found_syllable}", "", noise_part)
    if found_emotion:
        noise_part = re.sub(rf"(?i){found_emotion}", "", noise_part)
        
    noise_part = noise_part.strip("_")
    noise_part = re.sub(r"__+", "_", noise_part)
    
    if not noise_part or "clean" in noise_part.lower():
        noise_part = "clean"

    # 4. Classificar e gerar novo nome
    if found_syllable and not norm_emotion:
        # Etapa 1
        new_name = f"stage1_{found_syllable}_{noise_part}{ext}"
        return "Etapa 1", new_name
        
    elif found_syllable and norm_emotion:
        # Etapa 2
        new_name = f"stage2_{found_syllable}_{norm_emotion}_{noise_part}{ext}"
        return "Etapa 2", new_name
        
    elif norm_emotion and not found_syllable:
        # Etapa 3 Legada (Caso use arquivos legados apenas com emoções simples)
        digits = re.findall(r'\d+', cleaned_base)
        index = digits[0] if digits else "1"
        new_name = f"stage3_{norm_emotion}_{index}{ext}"
        return "Etapa 3 (Legado)", new_name
        
    return None, None

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERRO] A pasta de entrada '{INPUT_DIR}' não existe.")
        print("Crie-a e coloque seus vídeos originais dentro dela.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    extensions = ["*.mp4", "*.webm", "*.ogg", "*.mov"]
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(INPUT_DIR, ext)))
        
    if not files:
        print(f"Nenhum arquivo de vídeo encontrado na pasta '{INPUT_DIR}'.")
        return

    print(f"Iniciando varredura em: '{INPUT_DIR}'")
    if DRY_RUN:
        print("=== MODO DE TESTE (DRY RUN) ATIVADO ===")
        print("Nenhum arquivo será modificado na prática ainda.\n")

    success_count = 0
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        stage_detected, new_name = parse_and_rename(filename)
        
        if new_name:
            success_count += 1
            print(f"Original: {filename}")
            print(f"  -> Destino ({stage_detected}): {new_name}")
            
            if not DRY_RUN:
                dest_path = os.path.join(OUTPUT_DIR, new_name)
                # Copia em vez de mover para manter seus arquivos originais protegidos
                shutil.copy2(filepath, dest_path)
        else:
            print(f"[X] Ignorado (padrão não reconhecido): {filename}")
            
    print("\n------------------------------------------------")
    print(f"Varredura concluída. Arquivos válidos identificados: {success_count}/{len(files)}")
    if DRY_RUN:
        print("\nPara executar as alterações reais, abra o script e mude 'DRY_RUN = True' para 'DRY_RUN = False'.")

if __name__ == "__main__":
    main()