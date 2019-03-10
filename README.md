# GeoNet-Kitty
GeoNet在Kitty上的深度预测


### 1.DepthTask

记得先到Kitty_raw_loader里去改        

self.date_list = ['2011_09_26','2011_09_30'] #这里记录这要训练的数据集的地址

#### (1)数据预处理：  

我的代码是这样的 

> python data/prepare_train_data.py --dataset_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/kitty/ --dataset_name=kitti_raw_eigen --dump_root=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/kitty_formate/ --seq_length=3 --img_height=128 --img_width=416 --num_threads=1 --remove_static


预处理的数据不能是在data/kitti/test_scenes.txt里的文件

如果出现多线程的错误 记得把 num_threads=16 改成1

如果出现找不到2011_09_26/calib_cam_to_cam.txt的文件 那说明你少下了一个 2011_09_26_calib的文件 这里记录了当天使用的相机的具体参数

#### (2)训练

> python geonet_main.py --mode=train_rigid --dataset_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/kitty_formate/ --checkpoint_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/save_ckpts/ --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=11 --max_to_keep=2 --save_ckpt_freq=5



max_to_keep 是每多少步保存一次

他的代码geo_main.py里面缺了很多flags的声明

```python

flags.DEFINE_integer("num_source",                   2,   "His code loss this ")
flags.DEFINE_integer("num_scales",                   4,   "His code loss this ")
flags.DEFINE_string("add_flownet",                         "",    "His code loss this ")
flags.DEFINE_string("add_dispnet",                         "",    "His code loss this ")
flags.DEFINE_string("add_posenet",                         "",    "His code loss this ")

```

看这个issue：https://github.com/yzcjtr/GeoNet/pull/29

#### (3)测试

> python geonet_main.py --mode=test_depth --dataset_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/kitty/ --init_ckpt_file=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/save_ckpts/model-10 --batch_size=1 --depth_test_split=eigen --output_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/output_prediect/



test集合要改test_file_**文件  我这里有个问题 不知道为什么测试集只能25张图  会报错说文件读取失败

好像是有些图片的后缀是jpg不是png所以错了  对，他的test_file_**文件里 清一色的写的全是png 但是实际上数据集有jpg的



> python kitti_eval/eval_depth.py --split=eigen --kitti_dir=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/kitty/ --pred_file=/home/hu/Common/GeoNet-Kitty/GeoNet/data/Data/output_prediect/model-10.npy




| abs_rel    | sq_rel   |  rms  |  log_rms |  d1_all   |  a1   |  a2   |   a3  |
| --------   | -----:  | -----:  |-----:  |-----:  |-----:  |-----:  |:----:  |
| 0.4311    | 3.7952   |  10.1014  |  0.5420 |  0.0000   |  0.3240   |  0.5889   |   0.7995  |



GEOnet代码没有提供展示一张视差图结果图的方法 参考monodepth源码的monodepth——simple.py代码发现
在geonet_test_depth.py添加

```python

        disp_to_img = scipy.misc.imresize(pred_all[0].squeeze(), [opt.img_height, opt.img_width])
    	plt.imsave(os.path.join(opt.output_dir, "{}_disp.png".format("test")), disp_to_img, cmap='plasma')

```
即可展示出视差

#### 初步结果


![cmd-markdown-logo](./pic/1.png)

![cmd-markdown-logo](./pic/2.png)





