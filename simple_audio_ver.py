import jiwer
import argparse
import os
from normalizers.english import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

def get_wer(recos, transcriptions):
    if isinstance(recos, str):
        recos = [recos]
    if isinstance(transcriptions, str):
        transcriptions = [transcriptions]
        
    normalized_trans = [
        # jiwer doesn't score empty reference. Replace with ~ token - trading an insertion for substitution means WER same
        '~' if not x.strip() else x 
        for x in [normalizer(text) for text in transcriptions]
    ]
    normalized_reco = [ normalizer(text) for text in recos]
    wer = jiwer.wer(normalized_trans, normalized_reco)
    return wer*100

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./quant_onnx_models", help="quantized model directory")
parser.add_argument("--test_data_dir", default="./data/simple", help="test data directory")
args = parser.parse_args()

# load ground truth transcription. Note this is for full audio file, not the 30s chunks.
# we will run inference on 30 second chunks and just combine the text to score it
with open(os.path.join(args.test_data_dir, "transcript.txt"), 'r', encoding='utf-8') as f:
    lines = f.readlines()
trans = lines[0].split("\t")[1]



# =========
# Run inference on whisper model
# code below uses hugging face transformers, you could use OpenAI whisper version as well (see commented block below)
# =========

from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

#whisper_model_name = "openai/whisper-large-v2"  # NOTE: Use this for final benchmarking
whisper_model_name = "openai/whisper-tiny.en"
processor = AutoProcessor.from_pretrained(whisper_model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name)

#model = torch.quantization.quantize_dynamic(
#    model, {torch.nn.Linear}, dtype=torch.qint8
#)

device = 'cpu'
#device = 'cuda:0'
generator = pipeline(
    task="automatic-speech-recognition",
    model=model.to(device),
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    device=device,
    chunk_length_s=30,
    stride_length_s=(1,1) # Control level of chunk overlap
)

#save_directory_quantized = "./quantized_onnx_models"

# Load ORT from Disk
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import onnxruntime

session_options = onnxruntime.SessionOptions()
#session_options.enable_mem_pattern = False
#session_options.enable_cpu_mem_arena = False
model_ort_from_disk = ORTModelForSpeechSeq2Seq.from_pretrained(args.model_dir, session_options = session_options)
generator = pipeline(
    task="automatic-speech-recognition",
    model=model_ort_from_disk,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    # batch_size=2,
    # device=0,
    chunk_length_s=30,
    stride_length_s=(0,0), # must have with chunk_length_s
    #generate_kwargs={"max_new_tokens": 1024},
)

res1  = generator(os.path.join(args.test_data_dir, "PROG_SportsC-part-30s-1.wav"), generate_kwargs={"max_new_tokens": 5120})
res2  = generator(os.path.join(args.test_data_dir, "PROG_SportsC-part-30s-2.wav"), generate_kwargs={"max_new_tokens": 5120})
res3  = generator(os.path.join(args.test_data_dir, "PROG_SportsC-part-30s-3.wav"), generate_kwargs={"max_new_tokens": 5120})
partreco = f"{res1['text']} {res2['text']} {res3['text']}"
print("WER from HF impl:", get_wer(partreco, trans))