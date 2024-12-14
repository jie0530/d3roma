<h2 align="center">
  <b>D<sup>3</sup>RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation</b>

  <b><i>CoRL 2024, Munich, Germany.</i></b>


<div align="center">
    <a href="https://arxiv.org/abs/2409.14365" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://pku-epic.github.io/D3RoMa/" target="_blank">
    <img src="https://img.shields.io/badge/Page-D3RoMa-blue" alt="Project Page"/></a>
    <a href="https://openreview.net/forum?id=7E3JAys1xO" target="_blank">
    <img src="https://img.shields.io/badge/Page-OpenReview-blue" alt="Open Review"/></a>
</div>
</h2>

This is the official repository of [**D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation**](https://arxiv.org/abs/2409.14365).

For more information, please visit our [**project page**](https://pku-epic.github.io/D3RoMa/).

[Songlin Wei](https://songlin.github.io/),
[Haoran Geng](https://geng-haoran.github.io/),
[Jiayi Chen](https://jychen18.github.io/),
[Congyue Deng](https://cs.stanford.edu/~congyue/),
[Wenbo Cui](#),
[Chengyang Zhao](https://chengyzhao.github.io/)
[Xiaomeng Fang](#)
[Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)
[He Wang](https://hughw19.github.io/)


 
## 💡 Updates (Dec 14, 2024)

 - [x] We just release new model variant (Cond. on RGB+Raw), please checkout the updated inference.py
 - [ ] Traning protocols and dataset



Our method robustly predicts transparent (bottles) and specular (basin and cups) object depths in tabletop environments and beyond.
![teaser](assets/in-the-wild.png)



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

+ For model variant: Cond. Left+Right+Raw [Google drive](https://drive.google.com/file/d/12BLB7mKDbLPhW2UuJSmYnwBFokOjDvC9/view?usp=sharing), [百度云](https://pan.baidu.com/s/1u7n4wstGpqwAswp8ZbTNlw?pwd=o9nk)
+ For model variant: Cond. RGB+Raw [Google drive](https://drive.google.com/file/d/1cTAUZ2lXBXe4-peHLUneJ6ufQTqFr6E9/view?usp=drive_link), [百度云](https://pan.baidu.com/s/1zWwdMQ2_6-CViaC2JUGsFA?pwd=bwwb)
```
# Download pretrained weigths from Google Drive
# Extract it under the project folder
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

#### Contact
If you have any questions please contact us:

Songlin Wei: slwei@stu.pku.edu.cn, Haoran Geng: ghr@berkeley.edu, He Wang: hewang@pku.edu.cn

## Citation
```
@inproceedings{
  wei2024droma,
  title={D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation},
  author={Songlin Wei and Haoran Geng and Jiayi Chen and Congyue Deng and Cui Wenbo and Chengyang Zhao and Xiaomeng Fang and Leonidas Guibas and He Wang},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024},
  url={https://openreview.net/forum?id=7E3JAys1xO}
}
```


## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png