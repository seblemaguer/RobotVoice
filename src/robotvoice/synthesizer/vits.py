import io
import requests
import numpy.typing as npt
from scipy.io import wavfile

from robotvoice.synthesizer.base import Synthesizer


class DistantVITS(Synthesizer):
    """Wrapper for the VITS synthesizer running on a distant server."""

    def __init__(self, url: str = "localhost", port: int = 5000):
        """Initialisation

        Parameters
        ----------
        url : str
            The URL of the VITS server
        port : int
            The port of the VITS server
        """

        self._url = f"http://{url}:{port}/synth"

    def synth(self, text: str) -> tuple[npt.NDArray, int]:
        """Synthesize the given text using VITS

        Parameters
        ----------
        text : str
            The text to synthesize

        Returns
        -------
        tuple[npt.NDArray, int]
            The audio in a numpy array and the sampling rate

        Raises
        ------
        Exception
            if an error happens during the synthesis (mainly
            propagating the error from VITS)
        """

        # Query the server
        r = requests.post(self._url, json={"text": text})

        # If error propagate it
        if r.headers['content-type'] == "text/html; charset=utf-8":
            raise Exception(r.content)

        # We should receive an audio (wav format for now)!
        if r.headers['content-type'] !=  "audio/wav":
            raise Exception(f"The content of the response should be a wav but it is {r.headers['content-type']}")

        # Retrieve the audio information
        sr, audio = wavfile.read(io.BytesIO(r.content))
        return audio, sr
