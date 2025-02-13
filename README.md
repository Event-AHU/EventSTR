## EventOCR

* **EventSTR: A Benchmark Dataset and Baselines for Event Stream based Scene Text Recognition**, Xiao Wang, Jingtao Jiang, Dong Li, Futian Wang, Lin Zhu, Yaowei Wang, Yongyong Tian, Jin Tang, arXiv Paper
  [[arXiv]()]


### Abstract 
Mainstream Scene Text Recognition (STR) algorithms are developed based on RGB cameras which are sensitive to challenging factors such as low illumination, motion blur, and cluttered backgrounds. In this paper, we propose to recognize the scene text using bio-inspired event cameras by collecting and annotating a large-scale benchmark dataset, termed EventSTR. It contains 9,928 high-definition (1280 * 720) event samples and involves both Chinese and English characters. We also benchmark multiple STR algorithms as the baselines for future works to compare. In addition, we propose a new event-based scene text recognition framework, termed SimC-ESTR. It first extracts the event features using a visual encoder and projects them into tokens using a Q-former module. More importantly, we propose to augment the vision tokens based on a memory mechanism before feeding into the large language models. A similarity-based error correction mechanism is embedded within the large language model to correct potential minor errors fundamentally based on contextual information. Extensive experiments on the newly proposed EventSTR dataset and two simulation STR datasets fully demonstrate the effectiveness of our proposed model. We believe that the dataset and algorithmic model can innovatively propose an event-based STR task and are expected to accelerate the application of event cameras in various industries.



### Surveys and Reviews 

* [[https://github.com/Event-AHU/OCR_Paper_List](https://github.com/Event-AHU/OCR_Paper_List)] 


### :collision: Update Log 
* :fire: [2025-01-24] *****  


### :dvd:  Dataset Download 

* **Download from Baidu Drive:**
```
URL：https://pan.baidu.com/s/1XN8MfK1cKrqaSOo3e2oD3A?pwd=2l7c     Code：2l7c
```

* **Download from DropBox:** 
```
https://www.dropbox.com/scl/fo/s31llbv7bshz2xj4mf2gm/AFP1AGDcSoY0mk-fcyfL7jw?rlkey=p25w7366lzex7qe3pdgz96ec4&st=afcymd0x&dl=0
```

### :hammer: Environment Configuration 
1.Creating conda environment
```
conda create -n bliva python=3.9
conda activate bliva
```
2.build from source
```
git clone https://github.com/Event-AHU/EventSTR
cd EventSTR
pip install -e .
```

### :hammer: Prepare Weight 
Our Vicuna version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_Vicuna). Download our model weight and specify the path in the model config [here](https://github.com/Event-AHU/EventSTR/blob/384d37bececfc166d32d40c6fcd0ce64e1e16573/bliva/configs/models/bliva_vicuna7b.yaml#L8C4-L8C53) at line 8.

The LLM we used is the v0.1 version from Vicuna-7B. To prepare Vicuna's weight, please refer to our instruction [here](https://github.com/mlpc-ucsd/BLIVA/blob/main/PrepareVicuna.md). Then, set the path to the vicuna weight in the model config file [here](https://github.com/Event-AHU/EventSTR/blob/384d37bececfc166d32d40c6fcd0ce64e1e16573/bliva/configs/models/bliva_vicuna7b.yaml#L21) at Line 21.

### :hammer: Training & Testing 
Training
```
bash SimC-ESTR.sh
```
Testing
```
python test_bleu.py
```

### :hammer: Test FLOPs, and Speed 


### :cupid: Acknowledgement 




### :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex

```

If you have any questions about these works, please feel free to leave an issue. 



### Star History
