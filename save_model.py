# ORT:
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

#whisper_model_name = "openai/whisper-large-v2"  # NOTE: Use this for final benchmarking
whisper_model_name = "openai/whisper-tiny.en"

model_ort = ORTModelForSpeechSeq2Seq.from_pretrained(whisper_model_name, from_transformers=True, use_io_binding=True)
model_ort.save_pretrained("./onnx_models")