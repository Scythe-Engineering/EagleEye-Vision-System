# Training your own model
To train your model you first have to get your dataset, I recommend using roboflow to annotate and gather your dataset. Once you have your dataset you should train a model by running train.py, if you are just starting training start with yolov8n.pt (you can choose between 'n' 'm' or 's', n is the smallest and fastest) If you are further training a model that you already have, provide the path to it.
```bash
python3 train.py`
```
To export your model to the ONNX format that is much faster on cpu's run export.py
```bash
python3 export.py
```