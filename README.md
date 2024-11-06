#### INSTLLATION 
```
conda create --name d3roma python=3.8
conda activate d3roma

# install dependencies with pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install huggingface_hub==0.24.5
pip install diffusers opencv-python scikit-image matplotlib transformers datasets accelerate tensorboard imageio open3d kornia
pip install hydra-core --upgrade
```


#### DOWNLOAD PRE-TRAINED WEIGHT
```
https://drive.google.com/file/d/12BLB7mKDbLPhW2UuJSmYnwBFokOjDvC9/view?usp=sharing

Extract it under the project folder
```

#### RUN INFERENCE
You can run the following script to test our model:
```
python inference.py
```
This will generate three files under folder `_output`: 

`_outputs/pred.png`: the pseudo colored depth map

`_outputs/pred.ply`: the pointcloud which ia obtained though back-projected the predicted depth

`_outputs/raw.ply`: the pointcloud which ia obtained though back-projected the camera raw depth


#### Training Protocols & Dataset (Comming Soon)