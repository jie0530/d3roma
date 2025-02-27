
## Data Generation in simulation 

Although we do not plan to release all the sources for generating `HISS` dataset, I want to share an example code of generating IR renderings using [IsaacSim 4.0.0](https://docs.isaacsim.omniverse.nvidia.com/4.0.0/installation/install_container.html).

> This code should also work on Newer version of Isaac Sim with very few changes. If you encounter any problem please feel free to contact me.


### 1. prepare data

+ Download [HSSD scenes](https://huggingface.co/datasets/hssd/hssd-scenes) from here

    Notice that HSSD scenes are very big, you can download some of them for using.
    
    eg., I set [107734119_175999932](https://huggingface.co/datasets/hssd/hssd-scenes/blob/main/scenes/107734119_175999932.glb) as the default scene in `config/hssd.yaml`

    Please first convert it to USD file using [USD composer](https://docs.omniverse.nvidia.com/composer/latest/index.html).

+ Download object cad models from dreds, [link](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/cad_model/)

+ Download NVIDIA Omniverse [vMaterials_2](https://developer.nvidia.com/vmaterials)


Put them all in `data` folder, example folder structure:

```
data
├── dreds
│   ├── cad_model
│   │   ├── 00000000
│   │   ├── 02691156
│   │   ├── 02876657
│   │   ├── 02880940
│   │   ├── 02942699
│   │   ├── 02946921
│   │   ├── 02954340
│   │   ├── 02958343
│   │   ├── 02992529
│   │   └── 03797390
│   └── output
├── hssd
│   └── scenes
│       └── 107734119_175999932
└── vMaterials_2
    ├── Carpet
    .....
```

### 2. start isaac sim 4.0.0 Container

Change your project dir and start isaac-sim container

```
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/workspace/projects/d3roma/isaacsim:/root/d3roma:rw \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:4.0.0
```

### 3. install python packages into isaac-sim

```
/isaac-sim/python.sh -m pip install -r requirements.txt
```

### 4. generate IR renderings
```
/isaac-sim/python.sh render.py
```



