# Dialectal Arabic Tools
Dialectal Arabic Tools comprises the different modules developed in Qatar Computing Research Institute (QCRI) developed by the ALT team to handle Dialectal Arabic Segmentation, POS tagging, Diacritization and more

Dialectal Arabic Tools is compatible with: __Python 2.7-3.5 and later__.
## Prerequisites

Before you can use the dialectal Arabic tools you need to install a special version of keras that comprises a CRF layer. Use the following pash command to install it.

It is better to do installations within a virtual environment.
```sh
pip install git+git://github.com/phipleg/keras@crf
```

## Installation

You can install Dialectal Arabic Tools by either,
* using pip (recommended)
* cloning "this" repo and and use setup.py


### Installing Dialectal Arabic Tools via pip
Use the following pash command to install the package from the python index,
```sh
pip install dialectal_arabic_tools
```

### Installing Dialectal Arabic Tools from github
Clone the repo from the github website using the following command:
```sh
git clone https://github.com/qcri/dialectal_arabic_tools.git
```
Or download the compressed file of the project, extract it, change to the directory and run the following to install the Dialectal Arabic Tools using the following command:
```sh
 python setup.py install
```

## Getting started
Dialectal Arabic Tools package is pretty easy to use. The following code snippet uses the dialectal segmention module to module a string of Arabic script encoded in ``UTF-8``,
```python
>>> from dialectal_arabic_tools import segmentation
>>> segmentation.segment_text(u"عنا تنتين بندورة جبلية وخمسة عروقة نعنع بيعملو سلطة .. شلوني معك؟")
'عنا تنتين بندور+ة جبلي+ة و+خمس+ة عروق+ة نعنع ب+يعمل+و سلط+ة شلون+ي مع+ك ؟'
```

Furthermore, you could use the segmentation module to segment a text file of Arabic script encoded in ``UTF-8``. Just use ``segment_file`` insted of ``segment_text``.
The ``segment_file`` function requires two two positional parameters, namely the file to be segmented and a file name to generate the output in.

```python
>>> from dialectal_arabic_tools import segmentation
>>> segmentation.segment_file(r'/path/to/text/file/you/need/to/segment.txt', r'output/file/path.txt')
```


## Publications
Younes Samih, Mohamed Eldesouki, Mohammed Attia, Kareem Darwish, Ahmed Abdelali, Hamdy Mubarak, Laura Kallmeyer, (2017), [Learning from Relatives: Unified Dialectal Arabic Segmentation](http://www.aclweb.org/anthology/K17-1043), Journal Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), Pages 432-441.

Mohamed Eldesouki, Younes Samih, Ahmed Abdelali, Mohammed Attia, Hamdy Mubarak, Kareem Darwish, Kallmeyer Laura, (2017), [Arabic Multi-Dialect Segmentation: bi-LSTM-CRF vs. SVM](https://arxiv.org/pdf/1708.05891.pdf), arXiv preprint arXiv:1708.05891.

Younes Samih, Mohammed Attia, Mohamed Eldesouki, Ahmed Abdelali, Hamdy Mubarak, Laura Kallmeyer, Kareem Darwish, (2017), [A Neural Architecture for Dialectal Arabic Segmentation](http://www.aclweb.org/anthology/W17-1306), Journal Proceedings of the Third Arabic Natural Language Processing Workshop, Pages 46-54.





## Support

You can ask questions and join the development discussion:

- On the [Dialectal Arabic Tools Google group](https://groups.google.com/forum/#!forum/dat-users).
- On the [Dialectal Arabic Tools Slack channel](https://datsteam.slack.com). Use [this link](https://dat-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [Github issues](https://github.com/fqcri/dialectal_arabic_tools/issues). Make sure to read [our guidelines](https://github.com/qcri/dialectal_arabic_tools/blob/master/CONTRIBUTING.md) first.


------------------
