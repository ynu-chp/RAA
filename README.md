## Introduction


## Requirements


* Python >= 3.7
* Pytorch 1.10.0
* Argparse
* Logging
* Tqdm
* Scipy

### Generate the data

```bash
python generate_data.py
#You can set the number of user
#You can set the number of roi
```

### Train 

```python
#roi=5
# user=2
python 2x5.py

# user=4
python 4x5.py

# user=6
python 6x5.py

# user=8
python 8x5.py

# user=10
python 10x5.py

# user=12
python 12x5.py
```

```python
#user=5
# roi=2
python 5x2.py

# roi=4
python 5x4.py

# roi=6
python 5x6.py

# roi=8
python 5x8.py

# roi=10
python 5x10.py

# roi=12
python 5x12.py
```



## Acknowledgement

Our code is built upon the implementation of <https://arxiv.org/abs/2305.12162>.

