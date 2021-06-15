from tqdm import tqdm
from time import sleep

no_len = range(0, 50)
pbar = tqdm(no_len, total=50)

for i in range(50):
    pbar.update(1)
    pbar.write("Hello")
    sleep(1)