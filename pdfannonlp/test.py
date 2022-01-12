import time
from pathlib import Path
from extractor import Extractor


def main():
    config = {
        "data_dir": "/data/ling-sentence-extraction/test",
        "model_path": "/data/ling-sentence-extraction/model.pkl",
    }
    ex = Extractor.load(config["model_path"])
    ex.predict(config)


start = time.time()
main()
print(time.time() - start)
