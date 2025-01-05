link all the datasets here, example folder structures:

```
datasets
├── clearpose -> /raid/songlin/Data/clearpose
│   ├── clearpose_downsample_100
│   │   ├── downsample.py
│   │   ├── model
│   │   ├── set1
│   │   ├── set2
│   │   ├── set3
│   │   ├── set4
│   │   ├── set5
│   │   ├── set6
│   │   ├── set7
│   │   ├── set8
│   │   └── set9
│   ├── metadata
│   │   ├── set1
│   │   ├── set2
│   │   ├── set3
│   │   ├── set4
│   │   ├── set5
│   │   ├── set6
│   │   ├── set7
│   │   ├── set8
│   │   └── set9
│   ├── model
│   │   ├── 003_cracker_box
│   │   ├── 005_tomato_soup_can
│   │   ├── 006_mustard_bottle
│   │   ├── 007_tuna_fish_can
│   │   ├── 009_gelatin_box
│   │   ├── BBQSauce
│   │   ├── beaker_1
│   │   ├── bottle_1
│   │   ├── bottle_2
│   │   ├── bottle_3
│   │   ├── bottle_4
│   │   ├── bottle_5
│   │   ├── bowl_1
│   │   ├── bowl_2
│   │   ├── bowl_3
│   │   ├── bowl_4
│   │   ├── bowl_5
│   │   ├── bowl_6
│   │   ├── container_1
│   │   ├── container_2
│   │   ├── container_3
│   │   ├── container_4
│   │   ├── container_5
│   │   ├── create_keypoints.py
│   │   ├── dropper_1
│   │   ├── dropper_2
│   │   ├── flask_1
│   │   ├── fork_1
│   │   ├── funnel_1
│   │   ├── graduated_cylinder_1
│   │   ├── graduated_cylinder_2
│   │   ├── knife_1
│   │   ├── knife_2
│   │   ├── Mayo
│   │   ├── mug_1
│   │   ├── mug_2
│   │   ├── OrangeJuice
│   │   ├── pan_1
│   │   ├── pan_2
│   │   ├── pan_3
│   │   ├── pitcher_1
│   │   ├── plate_1
│   │   ├── plate_2
│   │   ├── reagent_bottle_1
│   │   ├── reagent_bottle_2
│   │   ├── round_table
│   │   ├── spoon_1
│   │   ├── spoon_2
│   │   ├── stick_1
│   │   ├── syringe_1
│   │   ├── trans_models.blend
│   │   ├── trans_models_keypoint.blend
│   │   ├── trans_models_keypoint.blend1
│   │   ├── trans_models_keypoint (copy).blend
│   │   ├── trans_models_kp.blend
│   │   ├── water_cup_1
│   │   ├── water_cup_10
│   │   ├── water_cup_11
│   │   ├── water_cup_12
│   │   ├── water_cup_13
│   │   ├── water_cup_14
│   │   ├── water_cup_2
│   │   ├── water_cup_3
│   │   ├── water_cup_4
│   │   ├── water_cup_5
│   │   ├── water_cup_6
│   │   ├── water_cup_7
│   │   ├── water_cup_8
│   │   ├── water_cup_9
│   │   ├── wine_cup_1
│   │   ├── wine_cup_2
│   │   ├── wine_cup_3
│   │   ├── wine_cup_4
│   │   ├── wine_cup_5
│   │   ├── wine_cup_6
│   │   ├── wine_cup_7
│   │   ├── wine_cup_8
│   │   └── wine_cup_9
│   ├── set1
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   └── scene5
│   ├── set2
│   │   ├── scene1
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   ├── set3
│   │   ├── scene1
│   │   ├── scene11
│   │   ├── scene3
│   │   ├── scene4
│   │   └── scene8
│   ├── set4
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   ├── set5
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   ├── set6
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   ├── set7
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   ├── set8
│   │   ├── scene1
│   │   ├── scene2
│   │   ├── scene3
│   │   ├── scene4
│   │   ├── scene5
│   │   └── scene6
│   └── set9
│       ├── scene10
│       ├── scene11
│       ├── scene12
│       ├── scene7
│       ├── scene8
│       └── scene9
├── DREDS
│   ├── test -> /raid/songlin/Data/DREDS_ECCV2022/DREDS-CatKnown/test
│   │   └── shapenet_generate_1216_val_novel
│   ├── test_std_catknown -> /raid/songlin/Data/DREDS_ECCV2022/STD-CatKnown
│   │   ├── test_0
│   │   ├── test_14-1
│   │   ├── test_18-1
│   │   ├── test_19
│   │   ├── test_20-3
│   │   ├── test_3-2
│   │   ├── test_4-2
│   │   ├── test_5-2
│   │   ├── test_6-1
│   │   ├── test_7-1
│   │   ├── test_8
│   │   ├── test_9-2
│   │   ├── train_0-5
│   │   ├── train_10-1
│   │   ├── train_12
│   │   ├── train_1-4
│   │   ├── train_14-1
│   │   ├── train_16-2
│   │   ├── train_17-1
│   │   ├── train_19-1
│   │   ├── train_3
│   │   ├── train_4-1
│   │   ├── train_7-1
│   │   ├── train_8
│   │   └── train_9-3
│   ├── test_std_catnovel -> /raid/songlin/Data/DREDS_ECCV2022/STD-CatNovel
│   │   └── real_data_novel
│   ├── train -> /raid/songlin/Data/DREDS_ECCV2022/DREDS-CatKnown/train
│   │   ├── part0
│   │   ├── part1
│   │   ├── part2
│   │   ├── part3
│   │   └── part4
│   └── val -> /raid/songlin/Data/DREDS_ECCV2022/DREDS-CatKnown/val
│       └── shapenet_generate_1216
├── HISS
│   ├── train -> /raid/songlin/Data/hssd-isaac-sim-100k
│   │   ├── 102344049
│   │   ├── 102344280
│   │   ├── 103997586_171030666
│   │   ├── 107734119_175999932
│   │   └── bad_his.txt
│   └── val -> /raid/songlin/Data/hssd-isaac-sim-300hq
│       ├── 102344049
│       ├── 102344280
│       ├── 103997586_171030666
│       ├── 107734119_175999932
│       ├── bad_his.txt
│       └── simulation2
├── README.md
├── Real
│   └── xiaomeng
│       ├── 0000_depth.png
│       ├── 0000_ir_l.png
│       ├── 0000_ir_r.png
│       ├── 0000_raw_disparity.png
│       ├── 0000_rgb.png
│       └── intrinsics.txt
└── sceneflow -> /raid/songlin/Data/sceneflow
    ├── bad_sceneflow_test.txt
    ├── bad_sceneflow_train.txt
    ├── Driving
    │   ├── disparity
    │   ├── frames_cleanpass
    │   ├── frames_finalpass
    │   ├── raw_cleanpass
    │   └── raw_finalpass
    ├── FlyingThings3D
    │   ├── disparity
    │   ├── frames_cleanpass
    │   ├── frames_finalpass
    │   ├── raw_cleanpass
    │   └── raw_finalpass
    └── Monkaa
        ├── disparity
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── raw_cleanpass
        └── raw_finalpass

227 directories, 18 files

```
