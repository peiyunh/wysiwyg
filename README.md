![Demo result](https://raw.githubusercontent.com/peiyunh/wysiwyg/master/demo.jpg)

# What You See Is What You Get:<br/>Exploiting Visibility for 3D Object Detection
By Peiyun Hu, Jason Ziglar, David Held, and Deva Ramanan

## Citing us
You can find our paper on [CVF Open Access](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.pdf). If you find our work useful, please consider citing:
```
@inproceedings{hu20wysiwyg,
  title={What You See Is What You Get: Exploiting Visibility for 3d Object Detection},
  author={Hu, Peiyun and Ziglar, Jason and Held, David and Ramanan, Deva},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11001--11009},
  year={2020}
}
```

## Setup
The code is developed based on [SECOND](https://github.com/traveller59/second.pytorch), a well-known deep network based 3D object detector. Please refer to this [README](https://github.com/traveller59/second.pytorch/blob/master/README.md) and [NUSCENES-GUIDE](https://github.com/traveller59/second.pytorch/blob/master/NUSCENES-GUIDE.md) on how to set up a working environment for the SECOND detector. 

## Pre-trained models
Download the pre-trained weights of the proposed model (early fusion) from [Google Drive](https://drive.google.com/file/d/1PeS6KCwi9JJlWG55MgEmNUHlgpkHeBHy/view?usp=sharing). The model configuration is located at [second/configs/nuscenes/all.pp.mhead.vpn.config](https://github.com/peiyunh/wysiwyg/blob/master/second/configs/nuscenes/all.pp.mhead.vpn.config). 

## Training
Run the following command to start training our model. 
```
python script.py train_nuscenes --base_cfg="all.pp.mhead.vpn.config" --sample_factor=1 --epochs=20 --eval_epoch=2 --sweep_db=True --label=vp_pp_oa_ta_learn --resume=True
```

## Raycasting
You can find the code for raycasting under [second/utils/mapping](https://github.com/peiyunh/wysiwyg/blob/master/second/utils/mapping/). The default setting for raycasting `drilling`.

We call raycasting functions in [second/data/preprocess.py](https://github.com/peiyunh/wysiwyg/blob/master/second/data/preprocess.py#L341-L359). If you are interested in integrating visibility into your own work, you can use it as an example of how to extract a visibility volume out of a LiDAR point cloud. 

You will need to compile the code using CMake. 
