import io
import json
import requests
import numpy as np
from scipy.io import wavfile


class VITS:
    def __init__(self, host: str = "localhost", port: int = 5000):
        self._url = f"http://{host}:{port}/synth"

    def synth(self, text: str) -> tuple[np.array, int]:

        r = requests.post(self._url, json={"text": text})
        print(r.headers['content-type'])
        if r.headers['content-type'] == "text/html; charset=utf-8":
            raise Exception(r.content)
        elif r.headers['content-type'] !=  "audio/wav":
            raise Exception(f"The content of the response should be a wav but it is {r.headers['content-type']}")
        sr, audio = wavfile.read(io.BytesIO(r.content))
        return audio, sr
