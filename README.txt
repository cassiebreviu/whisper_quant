1. install conda env:

    $ conda env create -f whisper.yml
    $ conda activate whisper

2. clone and install optimum:

    $ git clone --recursive https://github.com/yufenglee/optimum
    $ cd optimum
    $ git checkout whisper_static_quant
    $ python -m pip install -e .

3. insall onnxruntime:

    $ pip install onnxruntime

4. go to your work directory and download and save fp32 whisper model:
    
    $ python save_model.py

5. install ffmpeg

    $ conda install ffmpeg

6. quantize model with "quant.sh"

    $ bash quant.sh

7. run and verify the accuracy:

    $ python simple_audio_wer.py
