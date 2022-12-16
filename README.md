
## Training process
+ Set up enviroment
```bash
cd DeblurGAN_pytorch
pip install -r requirements.txt
cd ../st-gcn
pip install -r requirements.txt
cd pytorchlight
python3 setup.py install
cd ../..
pip install -r requirements.txt
```

+ Extracting Skeleton
```bash
cd DeblurGAN_pytorch
python extract_skeleton.py
# or
./run.sh
```

+ Action Quality Accessment

```bash
cd st-gcn
# prepare data
# split data by data/spilt_dataset.py
# and gendata .npy by tools/fds_gendata.py
./prepare_data
# start_training
./run.sh
```


