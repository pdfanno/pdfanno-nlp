from collections import defaultdict


def tag2span(tags: list[str]) -> dict[str, list[tuple[int, int]]]:  # span label -> [(begin, end)]
    spans = {}  # tag -> (begin, end)
    s = -1
    for k, tag in enumerate(tags):
        if tag.startswith("S-"):
            if tag[2:] not in spans:
                spans[tag[2:]] = []
            spans[tag[2:]].append((k, k + 1))
            s = -1
        elif tag.startswith("B-"):
            s = k
        elif tag.startswith("I-") or tag.startswith("E-"):
            continue
        elif s >= 0:
            tag = tags[s][2:]
            if tag not in spans:
                spans[tag] = []
            spans[tag].append((s, k))
            s = -1
    if s >= 0:
        tag = tags[s][2:]
        if tag not in spans:
            spans[tag] = []
        spans[tag].append((s, len(tags)))
    return spans


def evaluate_span(pred_tags: list[str], gold_tags: list[str]):
    """
    スパンレベルでの評価
    """
    pred_spans = tag2span(pred_tags)
    gold_spans = tag2span(gold_tags)
    print("Tag\tPrecision\tRecall\tF1")
    for tag in gold_spans.keys():
        gold = {g for g in gold_spans[tag]}
        if tag not in pred_spans:
            pred = {}
        else:
            pred = {p for p in pred_spans[tag]}
        common = gold.intersection(pred)

        p = len(common) / (len(pred) + 1e-5)
        r = len(common) / (len(gold) + 1e-5)
        f = 2 * p * r / (p + r + 1e-5)
        print(f"{tag}\t{p:.3f}\t{r:.3f}\t{f:.3f}")
    print("")


def evaluate_bio(pred_tags: list[str], gold_tags: list[str]):
    """
    BIOタグでの評価
    """
    assert len(pred_tags) == len(gold_tags)
    counter = defaultdict(lambda: [0, 0, 0])  # gold, pred, common
    for y, z in zip(gold_tags, pred_tags):
        counter[y][0] += 1
        counter[z][1] += 1
        if y == z:
            counter[y][2] += 1

    print("Tag\tPrecision\tRecall\tF1")
    for tag, counts in counter.items():
        p = counts[2] / (counts[1] + 1e-5)
        r = counts[2] / (counts[0] + 1e-5)
        f1 = 2 * p * r / (p + r + 1e-5)
        print(f"{tag}\t{p:.3f}\t{r:.3f}\t{f1:.3f}")
    print("")
