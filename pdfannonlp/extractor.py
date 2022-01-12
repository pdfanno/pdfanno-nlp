import json

import cloudpickle
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pdf import PDF
import metrics
from rect import Rect
from ner.dataset import Dataset
from ner.models import Tagger


class Extractor:

    def __init__(self):
        self.model = None
        self.words = None
        self.tags = None
        self.token2id = None

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            extractor = cloudpickle.load(f)
        return extractor

    def save(self, filename):
        # state_dict = self.model.to("cpu").state_dict()
        # torch.save(state_dict, filename)
        self.model = self.model.to("cpu")
        with open(filename, 'wb') as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def train(config):
        extractor = Extractor()
        # words.extend(["PAD", "UNK"])  # "􅨣" is unnecessary
        # parser.words = words
        # word2id = {w: i for i, w in enumerate(words)}
        # embeds = torch.tensor(embeds, dtype=torch.float)
        # assert embeds.ndim == 2
        # m = torch.rand(len(words) - embeds.size(0) + 1, embeds.size(1))
        # embeds = torch.cat([embeds, m])

        pdf_files = list(Path(config["data_dir"]).glob("*.pdf"))
        print(f"Load {len(pdf_files)} pdf files...")
        pdfs = [PDF.load(pdf_file) for pdf_file in pdf_files]
        texts = []
        for pdf in pdfs:
            anno_file = Path(pdf.filename + ".anno")
            pdf.read_pdfanno(str(anno_file), config["input_labels"], config["tag_mapping"])
            for page in pdf.pages:
                texts.extend(page.texts)

        # tag dictionary
        tag2id = {}
        for text in texts:
            for w in text.words:
                if w.tag not in tag2id:
                    tag2id[w.tag] = len(tag2id)
        print(f"Tag dictionary: {tag2id.keys()}")
        tags = [-1 for _ in range(len(tag2id))]
        for t, i in tag2id.items():
            tags[i] = t
        extractor.tags = tags

        random.shuffle(texts)
        k = round(0.9 * len(texts))
        # k = len(texts)
        token2id = {"UNK": 0}
        extractor.token2id = token2id
        train_dataset = Dataset(texts[:k], tag2id, token2id, is_train=True)
        dev_dataset = Dataset(texts[k:], tag2id, token2id, is_train=False)
        # dev_dataset = None
        print(f"# of training texts: {k}")
        print(f"# of development texts: {len(texts) - k}")
        print(f"# of tokens: {len(token2id)}")
        # print(token2id.keys())

        # model
        model = Tagger(num_tokens=len(token2id), num_tags=len(tag2id))
        extractor.model = model

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # pdf.evaluate(dev_dataset)

        num_epochs = 50
        step = 100
        for epoch in range(1, num_epochs + 1):
            model.train()
            print(f"Epoch: {epoch}")
            start = time.time()
            running_loss = 0.0
            for i, instance in enumerate(train_loader, 1):
                optimizer.zero_grad()
                loss = model(instance)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % step == 0 or i == len(train_dataset):
                    print(f"[{epoch}, {i}] loss: {(running_loss / step):.3f}")
                    running_loss = 0.0
            print(f"Time: {time.time() - start}[sec]")

            if dev_dataset is not None:
                extractor.evaluate(dev_dataset)
            if epoch % 10 == 0 or epoch == num_epochs:
                extractor.save(config["model_path"])
            scheduler.step()
        return extractor

    def evaluate(self, dataset):
        model = self.model
        model.eval()
        pred_ids = []
        gold_ids = []
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for instance in data_loader:
                pred_id = model(instance).reshape(-1).cpu()
                pred_ids.extend(pred_id.tolist())
                gold_id = instance[-1].reshape(-1).cpu()
                gold_ids.extend(gold_id.tolist())

        pred_tags = [self.tags[id] for id in pred_ids]
        gold_tags = [self.tags[id] for id in gold_ids]
        metrics.evaluate_bio(pred_tags, gold_tags)
        metrics.evaluate_span(pred_tags, gold_tags)

    def predict(self, config):
        tag2id = {t: i for i, t in enumerate(self.tags)}
        model = self.model
        model.eval()

        pdf_files = list(Path(config["data_dir"]).glob("*.pdf"))
        for pdf_file in pdf_files:
            print(f"Loading {pdf_file.name}...")
            pdf = PDF.load(pdf_file)
            if len(pdf.pages) == 0:
                continue
            dataset = Dataset(pdf.pages, tag2id, self.token2id, is_train=False)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            k = 0
            anno_dict = []
            with torch.no_grad():
                for instance in data_loader:
                    anno_dict.append([])
                    pred_id = model(instance).reshape(-1).cpu()
                    pred_tags = [self.tags[id] for id in pred_id.tolist()]
                    spans_dict = metrics.tag2span(pred_tags)

                    page = pdf.pages[k]
                    for label, spans in spans_dict.items():
                        for s, e in spans:
                            bboxes = [w.bbox for w in page.words[s:e]]
                            bbox = Rect.from_rects(bboxes).clip(0, 1)
                            bbox = [round(x, 5) for x in bbox.list()]
                            anno_dict[-1].append({"label": label, "bbox": bbox})
                    k += 1
            with open(str(pdf_file) + ".anno", "w", encoding="utf-8") as f:
                json.dump(anno_dict, f)
