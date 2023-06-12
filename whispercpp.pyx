#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path
from libc.stdlib cimport malloc

MODELS_DIR = str(Path('~/.ggml-models').expanduser())
print("Saving models to:", MODELS_DIR)


cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef char* LANGUAGE = b'en'
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-base.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(model):
    return os.path.exists(Path(MODELS_DIR).joinpath(model))

def download_model(model):
    if model_exists(model):
        return

    print(f'Downloading {model}...')
    url = MODELS[model]
    r = requests.get(url, allow_redirects=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(Path(MODELS_DIR).joinpath(model), 'wb') as f:
        f.write(r.content)


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
        out = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le",
                acodec="pcm_s16le",
                ac=1, ar=sr
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True
            )
        )[0]
    except:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    return frames

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *> LANGUAGE
    n_threads = N_THREADS
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model=DEFAULT_MODEL, pb=None, buf=None):
        
        model_fullname = f'ggml-{model}.bin'
        download_model(model_fullname)
        model_path = Path(MODELS_DIR).joinpath(model_fullname)
        cdef bytes model_b = str(model_path).encode('utf8')
        
        if buf is not None:
            self.ctx = whisper_init_from_buffer(buf, buf.size)
        else:
            self.ctx = whisper_init_from_file(model_b)
        
        self.params = default_params()
        whisper_print_system_info()


    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE):
        print("Loading data..")
        if (type(filename) == np.ndarray) :
            temp = filename
        
        elif (type(filename) == str) :
            temp = load_audio(<bytes>filename)
        else :
            temp = load_audio(<bytes>TEST_FILE)

        
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = temp

        print("Transcribing..")
        return whisper_full(self.ctx, self.params, &frames[0], len(frames))
    
    def extract_text(self, int res):
        print("Extracting text...")
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_text(self.ctx, i).decode() for i in range(n_segments)
        ]
    
    def get_params(self):
        params = {}
        prompt_tokens = []
        
        # getting prompt_tokens
#         cdef int prompt_token_len = <int>(sizeof(self.params.prompt_tokens) / sizeof(int))
        cdef int i
        for i in range(self.params.prompt_n_tokens):
#             print(i)
#             print(<int>self.params.prompt_tokens[i])
            prompt_tokens.append(<int>self.params.prompt_tokens[i])
        
        # filling params dict
        params.update({
            "n_threads": self.params.n_threads,
            "n_max_text_ctx": self.params.n_max_text_ctx,
            "offset_ms": self.params.offset_ms,
            "duration_ms": self.params.duration_ms,
            "translate": self.params.translate,
            "no_context": self.params.no_context,
            "single_segment": self.params.single_segment,
            "print_special": self.params.print_special,
            "print_progress": self.params.print_progress,
            "print_realtime": self.params.print_realtime,
            "print_timestamps": self.params.print_timestamps,
            "token_timestamps": self.params.token_timestamps,
            "thold_pt": self.params.thold_pt,
            "thold_ptsum": self.params.thold_ptsum,
            "max_len": self.params.max_len,
            "max_tokens": self.params.max_tokens,
            "speed_up": self.params.speed_up,
            "audio_ctx": self.params.audio_ctx,
            "prompt_tokens": prompt_tokens,
            "prompt_n_tokens": self.params.prompt_n_tokens,
            "language": self.params.language
        })
        
        return params
    
    def set_params(self, options):
        cdef int *prompt_tokens_c
        
        if "n_threads" in options:
            self.params.n_threads = options["n_threads"]
        if "n_max_text_ctx" in options:
            self.params.n_max_text_ctx = options["n_max_text_ctx"]
        if "offset_ms" in options:
            self.params.offset_ms = options["offset_ms"]
        if "duration_ms" in options:
            self.params.duration_ms = options["duration_ms"]
        if "translate" in options:
            self.params.translate = options["translate"]
        if "no_context" in options:
            self.params.no_context = options["no_context"]
        if "single_segment" in options:
            self.params.single_segment = options["single_segment"]
        if "print_special" in options:
            self.params.print_special = options["print_special"]
        if "print_progress" in options:
            self.params.print_progress = options["print_progress"]
        if "print_realtime" in options:
            self.params.print_realtime = options["print_realtime"]
        if "print_timestamps" in options:
            self.params.print_timestamps = options["print_timestamps"]
        if "token_timestamps" in options:
            self.params.token_timestamps = options["token_timestamps"]
        if "thold_pt" in options:
            self.params.thold_pt = options["thold_pt"]
        if "thold_ptsum" in options:
            self.params.thold_ptsum = options["thold_ptsum"]
        if "max_len" in options:
            self.params.max_len = options["max_len"]
        if "max_tokens" in options:
            self.params.max_tokens = options["max_tokens"]
        if "speed_up" in options:
            self.params.speed_up = options["speed_up"]
        if "audio_ctx" in options:
            self.params.audio_ctx = options["audio_ctx"]
        if "prompt_tokens" in options:
            prompt_tokens_py = options["prompt_tokens"]
            arr_length = len(prompt_tokens_py)
            prompt_tokens_c = <int*>malloc(arr_length * sizeof(int))
            for i in range(arr_length):
                prompt_tokens_c[i] = prompt_tokens_py[i]
            self.params.prompt_tokens = prompt_tokens_c
            self.params.prompt_n_tokens = arr_length
        if "language" in options:
            self.params.language = options["language"]
