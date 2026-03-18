Useage: 

1.
conda create --name T2S python=3.9
source activate T2S
pip install requirements.txt

2.
The basicsr package may have incompatibility issues with the torchvision package (if torch>1.13), please modify the basicsr environment in '.conda' manually for the time being.
vim ~/.conda/envs/T2S/lib/python3.9/site-packages/basicsr/data/degradations.py (Depends on your environment)
Modify the line 8 to 'from torchvision.transforms._functional_tensor import rgb_to_grayscale'

3.
Arrange dataset as used in './Dataset.py' to the file './Dataset' 

4.
Download the SD v1.5, T2I-adapter, pretrained model for dlib, and pretrained model for pidinet to the file './models'
Dowload pretrained model from https://drive.google.com/file/d/1Z6XaPaP24RN-rzb7SrwyHYcyTseOysBz/view?usp=sharing 

5.
python train_partial_sketch.py
python train_face.py

test:
python -u user_base.py --scale 3.5 --data_path Custom_Dataset --model_path $MODEL_PATH 

