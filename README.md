1. install conda env "conda env create -f whisper.yml"
2. clone and install optimum
    git clone --recursive https://github.com/yufenglee/optimum
    cd optimum
    git checkout whisper_static_quant
    python -m pip install -e .
3. go to your work directory and download and save fp32 whisper model
    python save_model.py
4. quantize model with "quant.sh"
5. run and verify the accuracy
    python simple_audio_wer.py

