#!/usr/bin/env python3
# Modified version to handle long audio files with chunking and debugging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_audio(audio, sample_rate, chunk_duration=30, overlap_duration=0):
    """
    Split audio into chunks with optional overlap between consecutive chunks
    
    Args:
        audio: tensor of shape (channels, samples)
        sample_rate: sample rate of audio
        chunk_duration: duration of each chunk in seconds
        overlap_duration: overlap between consecutive chunks in seconds
    
    Returns:
        list of audio chunks
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step_samples = chunk_samples - overlap_samples
    num_samples = audio.shape[1]
    
    # If audio is shorter than chunk duration, return as single chunk
    if num_samples <= chunk_samples:
        return [audio]
    
    chunks = []
    start = 0
    
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        chunk = audio[:, start:end]
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start += step_samples
        
        # Stop if we've covered all audio
        if end >= num_samples:
            break
    
    return chunks

def single_job(utt):
    try:
        # Load audio
        audio, sample_rate = torchaudio.load(utt2wav[utt], backend='soundfile')
        logging.info(f"Processing {utt}: Original shape={audio.shape}, sample_rate={sample_rate}")
        
        # Resample if needed
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
            sample_rate = 16000
            logging.info(f"  Resampled to 16kHz: shape={audio.shape}")
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            logging.info(f"  Converted to mono: shape={audio.shape}")
        
        duration = audio.shape[1] / sample_rate
        logging.info(f"  Duration: {duration:.2f} seconds")
        
        # Split into chunks
        chunks = chunk_audio(
            audio, 
            sample_rate, 
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap_duration
        )
        logging.info(f"  Split into {len(chunks)} chunks")
        
        # Store tokens for each chunk separately
        chunk_tokens = {}
        
        for i, chunk in enumerate(chunks):
            chunk_duration = chunk.shape[1] / sample_rate
            chunk_start_time = i * (args.chunk_duration - args.overlap_duration)
            chunk_end_time = chunk_start_time + chunk_duration
            
            logging.info(f"  Processing chunk {i}/{len(chunks)-1}: time={chunk_start_time:.1f}s-{chunk_end_time:.1f}s, duration={chunk_duration:.2f}s, shape={chunk.shape}")
            
            # Extract mel spectrogram
            feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
            logging.info(f"    Mel spectrogram shape: {feat.shape}")
            
            # Run ONNX model
            onnx_input_feat = feat.detach().cpu().numpy()
            onnx_input_len = np.array([feat.shape[2]], dtype=np.int32)
            
            logging.info(f"    ONNX input feat shape: {onnx_input_feat.shape}")
            logging.info(f"    ONNX input len: {onnx_input_len}")
            
            speech_token = ort_session.run(
                None, 
                {
                    ort_session.get_inputs()[0].name: onnx_input_feat,
                    ort_session.get_inputs()[1].name: onnx_input_len
                }
            )[0]
            
            logging.info(f"    ONNX output shape: {speech_token.shape}")
            speech_token_list = speech_token.flatten().tolist()
            logging.info(f"    Number of tokens: {len(speech_token_list)}")
            logging.info(f"    Token range: [{min(speech_token_list)}, {max(speech_token_list)}]")
            logging.info(f"    First 10 tokens: {speech_token_list[:10]}")
            
            # Save chunk with index
            chunk_key = f"{utt}_chunk_{i}"
            chunk_tokens[chunk_key] = speech_token_list
        
        total_tokens = sum(len(tokens) for tokens in chunk_tokens.values())
        logging.info(f"  Total chunks for {utt}: {len(chunk_tokens)}, total tokens: {total_tokens}")
        return utt, chunk_tokens
        
    except Exception as e:
        logging.error(f"Error processing {utt}: {str(e)}")
        return utt, {}

def main(args):
    logging.info(f"Processing {len(utt2wav)} audio files")
    logging.info(f"Using {args.num_thread} threads")
    
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    
    for future in tqdm(as_completed(all_task), total=len(all_task)):
        utt, chunk_tokens = future.result()
        # Merge chunk tokens into main dictionary
        utt2speech_token.update(chunk_tokens)
    
    output_path = Path(args.output_dir) / 'utt2speech_token_test.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(utt2speech_token, str(output_path))
    logging.info(f"Saved speech tokens to {output_path}")
    
    # Print summary statistics
    total_chunks = len(utt2speech_token)
    total_tokens = sum(len(tokens) for tokens in utt2speech_token.values())
    avg_tokens = total_tokens / total_chunks if total_chunks else 0
    logging.info(f"\nSummary:")
    logging.info(f"  Total audio files: {len(utt2wav)}")
    logging.info(f"  Total chunks: {total_chunks}")
    logging.info(f"  Total tokens: {total_tokens}")
    logging.info(f"  Average tokens per chunk: {avg_tokens:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True, 
                        help="Directory containing wav.scp file")
    parser.add_argument("--onnx_path", type=str, default= "pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output (default: same as wav_dir)")
    parser.add_argument("--num_thread", type=int, default=8,
                        help="Number of threads for parallel processing")
    parser.add_argument("--chunk_duration", type=int, default=30,
                        help="Duration of each audio chunk in seconds")
    parser.add_argument("--overlap_duration", type=float, default=0.0,
                        help="Overlap between consecutive chunks in seconds")
    args = parser.parse_args()
    
    # Set output dir to wav_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.wav_dir
    
    # Load wav.scp
    utt2wav = {}
    wav_scp_path = Path(args.wav_dir)
    
    if not wav_scp_path.exists():
        logging.error(f"wav.scp not found at {wav_scp_path}")
        exit(1)
    
    with open(wav_scp_path) as f:
        for l in f:
            l = l.replace('\n', '').split()
            if len(l) >= 2:
                utt2wav[l[0]] = l[1]
    
    logging.info(f"Loaded {len(utt2wav)} utterances from {wav_scp_path}")
    
    # Setup ONNX Runtime
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    logging.info(f"Loading ONNX model from {args.onnx_path}")
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    
    # Print model info
    logging.info("ONNX Model Information:")
    for i, input_node in enumerate(ort_session.get_inputs()):
        logging.info(f"  Input {i}: name={input_node.name}, shape={input_node.shape}, dtype={input_node.type}")
    for i, output_node in enumerate(ort_session.get_outputs()):
        logging.info(f"  Output {i}: name={output_node.name}, shape={output_node.shape}, dtype={output_node.type}")
    
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    
    main(args)