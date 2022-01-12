# pdfanno-nlp

## Install
* Python 3.9以上
* [pytorch](https://pytorch.org/) 1.10以上
```
$ pip install cloudpickle pytorch-crf
```

## Training
`train.py`のconfig情報を環境に合わせて設定
```
$ cd src
$ python train.py
```
`data_dir`には，`.pdf`, `.pdf.txt.gz`, `.anno`ファイルが必要

## Testing
`test.py`のconfig情報を環境に合わせて設定
```
$ cd src
$ python test.py
```
`data_dir`には，`.pdf`, `.pdf.txt.gz`ファイルが必要．  
annoファイルが生成される．
