import onnx
import argparse
import os
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QDQQuantizer


from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import onnxruntime

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

whisper_model_name = "openai/whisper-tiny.en"

processor = AutoProcessor.from_pretrained(whisper_model_name)

class WhisperGeneration:
    def __init__(self,
                 model_dir,):
        self.model_dir = model_dir
        self.model_ort = ORTModelForSpeechSeq2Seq.from_pretrained(self.model_dir)
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model_ort,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            # batch_size=2,
            # device=0,
            chunk_length_s=30,
            stride_length_s=(1,1), # must have with chunk_length_s
            generate_kwargs={"max_new_tokens": 1024},)

        self.pipeline.model.encoder.collect_inputs = True
        self.pipeline.model.decoder.collect_inputs = True
        self.pipeline.model.decoder_with_past.collect_inputs = True

        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_with_past_inputs = None

    def generate(self, audio_path):
        self.pipeline.model.encoder.input_values = []
        self.pipeline.model.decoder.input_values = []
        self.pipeline.model.decoder_with_past.input_values = []
        _ = self.pipeline(audio_path)
        self.encoder_inputs = self.pipeline.model.encoder.input_values
        self.decoder_inputs = self.pipeline.model.decoder.input_values
        self.decoder_with_past_inputs = self.pipeline.model.decoder_with_past.input_values

class WhisperEncoderReader(CalibrationDataReader):
    def __init__(self, whisper_gen):
        self.whisper_gen = whisper_gen
        self.values = iter(self.whisper_gen.encoder_inputs)

    def get_next(self):
        return next(self.values, None)

class WhisperDecoderReader(CalibrationDataReader):
    def __init__(self, whisper_gen):
        self.whisper_gen = whisper_gen
        self.values = iter(self.whisper_gen.decoder_inputs)

    def get_next(self):
        return next(self.values, None)


class WhisperDecoderWithPastReader(CalibrationDataReader):
    def __init__(self, whisper_gen):
        self.whisper_gen = whisper_gen
        self.values = iter(self.whisper_gen.decoder_with_past_inputs)

    def get_next(self):
        return next(self.values, None)


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="encoder model of whisper")
parser.add_argument("--cali_query", required=True, help="Queries to calibrate")
parser.add_argument("--quant_model_dir", required=True, help="quantized model")
args = parser.parse_args()

model_names = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]
model_files = [os.path.join(args.model_dir, model_name) for model_name in model_names]
quant_model_files = [os.path.join(args.quant_model_dir, model_name) for model_name in model_names]
calibrators = [create_calibrator(model_file,
                                op_types_to_calibrate = ['MatMul', 'Gemm'],
                                calibrate_method=CalibrationMethod.MinMax,
                                use_external_data_format=True) for model_file in model_files]

readers = [WhisperEncoderReader, WhisperDecoderReader, WhisperDecoderWithPastReader]


whisper_gen = WhisperGeneration(args.model_dir)

audio_paths = open(args.cali_query, "r")
while True:
    audio_path = audio_paths.readline()
    if not audio_path:
        break

    print("start collect data for encoder")
    whisper_gen.generate(audio_path.strip())

    for i in range(len(model_names)):
        reader = readers[i](whisper_gen)
        calibrators[i].collect_data(reader)

compute_range = [calibrator.compute_range() for calibrator in calibrators]

op_types_to_quantize = ['MatMul', 'Gemm']

for i in range(len(model_names)):
    model = onnx.load_model(model_files[i])
    quantizer = QDQQuantizer(
        model,
        False, #per_channel
        False, #reduce_range
        QuantizationMode.QLinearOps,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range[i],
        [], #nodes_to_quantize
        None,
        op_types_to_quantize,
        {'ActivationSymmetric' : True,
         'MatMulConstBOnly' : True,
         'OpTypesToExcludeOutputQuantization' : op_types_to_quantize,
         'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} }) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(quant_model_files[i], True)