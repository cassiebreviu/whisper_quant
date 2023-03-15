#python whisper_quant.py \
#       --encoder_model /work/whisper-yufeng/onnx_models/encoder_model.onnx \
#       --decoder_model /work/whisper-yufeng/onnx_models/decoder_model.onnx \
#       --decoder_with_past_model /work/whisper-yufeng/onnx_models/decoder_with_past_model.onnx \
#       --model_dir /work/whisper-yufeng/onnx_models/ \
#       --cali_query /work/whisper-yufeng/cal_audios.txt \
#       --quant_encoder_model /work/whisper-yufeng/encoder_quant/model.onnx \
#       --quant_decoder_model /work/whisper-yufeng/decoder_quant/model.onnx \
#       --quant_decoder_with_past_model /work/whisper-yufeng/decoder_with_past_quant/model.onnx

mkdir -p ./quant_onnx_models
python whisper_quant.py \
       --model_dir ./onnx_models/ \
       --cali_query ./cal_audios.txt \
       --quant_model_dir ./quant_onnx_models

cp ./onnx_models/*.json ./quant_onnx_models