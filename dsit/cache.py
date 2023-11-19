import json
from pathlib import Path
import scipy.io.wavfile as wav

from dsit import CACHE_JSON, AUDIO_DIR, CACHE_DIR
from dsit.utils import get_json, create_folder_if_not_exists


class Cache:

    @staticmethod
    def get():
        try:
            return get_json(CACHE_JSON)
        except FileNotFoundError:
            return {}

    @staticmethod
    def get_by_key(**keys):
        return Cache.get()[Cache._build_key(**keys)]

    @staticmethod
    def set_by_key(result, **keys):
        create_folder_if_not_exists(CACHE_DIR)
        cache = Cache.get()

        cache[Cache._build_key(**keys)] = result

        with open(CACHE_JSON, "w+") as f:
            json.dump(cache, f)

    @staticmethod
    def key_exists(**keys):
        return Cache._build_key(**keys) in Cache.get().keys()

    def remove_cache(self):
        # TODO Remove the cache folder
        pass

    @staticmethod
    def _build_key(**keys) -> int:
        framerate, audio_signal = wav.read(Path(f"{AUDIO_DIR}{keys['name']}.wav"))

        print(hash(framerate), hash(audio_signal.tostring()), hash(tuple(sorted(keys.values()))))
        return hash((framerate, audio_signal.tostring(), tuple(sorted(keys.values()))))
