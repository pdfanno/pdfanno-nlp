# pdfanno-nlp

## Install
* Python 3.9 or higher
```
$ pip install ./pdfanno-nlp
```
If you modify the code:
```
$ pip install -e ./pdfanno-nlp
```

## Load PDF
```
from pdfannonlp import PDF, Char, Word

pdf = PDF.load_text("xxx.pdf.json.gz")
```