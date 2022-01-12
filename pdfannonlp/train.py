import time
from pathlib import Path
from extractor import Extractor


def main():
    config = {
        "data_dir": "/data/kato/kobunshi",
        "model_path": "/data/ling-sentence-extraction/model.pkl",
        "input_labels": ["original_composite_unit"],
    }
    Extractor.train(config)


start = time.time()
main()
print(time.time() - start)
