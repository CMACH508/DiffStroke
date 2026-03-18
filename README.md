# Usage

## A more detailed README.md file will be updated. The corresponding dataset will also be uploaded later.

## 1. Create the environment

```bash
conda create --name T2S python=3.9
conda activate T2S
pip install -r requirements.txt
```

## 2. Fix `basicsr` compatibility issue

The `basicsr` package may be incompatible with `torchvision` when `torch > 1.13`.

Please manually modify the `basicsr` file in your conda environment:

```bash
vim ~/.conda/envs/T2S/lib/python3.9/site-packages/basicsr/data/degradations.py
```

> The actual path may vary depending on your environment.

Then change line 8 to:

```python
from torchvision.transforms._functional_tensor import rgb_to_grayscale
```

## 3. Prepare the dataset

Arrange the dataset according to the format used in `./Dataset.py`, and place it under:

```bash
./Dataset
```

## 4. Download pretrained models

Download the following files into the `./models` directory:

- Stable Diffusion v1.5
- T2I-Adapter
- Pretrained model for `dlib`
- Pretrained model for `pidinet`

Pretrained model download link:

[Google Drive](https://drive.google.com/file/d/1Z6XaPaP24RN-rzb7SrwyHYcyTseOysBz/view?usp=sharing)

## 5. Train the model

```bash
python train_partial_sketch.py
python train_face.py
```

## Testing

```bash
python -u user_base.py --scale 3.5 --data_path Custom_Dataset --model_path $MODEL_PATH
```
