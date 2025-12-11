# local_inn

Prerequisites
- Python 3.11 or higher

```
source setup.bash
```

Move your rosbag to `data/`

Run
```
python dataloader.py <EXP_NAME> <ROSBAG_NAME>
python dataprocess.py <EXP_NAME>
python model.py <EXP_NAME>
```

