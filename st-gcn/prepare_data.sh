
cd data
python3 split.py
cd ..

data_path=data/FineDiving
python tools/fds_gendata.py --data_path ${data_path}