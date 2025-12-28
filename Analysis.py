import pandas as pd
import numpy as np
import ast  # 用來把字串 ('0.1','0.2','0.3') 轉成 tuple

df = pd.read_csv("track_2.csv")

# 取 p0 到 p32，解析成 numpy array (frames, joints, 3)
joints = []
for col in df.columns[1:]:  # 跳過 frame 欄
    coords = df[col].apply(lambda s: np.array(ast.literal_eval(s), dtype=float))
    joints.append(np.stack(coords.values))

# joints: list of (frames, 3)
joints = np.stack(joints, axis=1)  # shape = (frames, 33, 3)

print("骨架資料形狀:", joints.shape)
# => (2657, 33, 3)
