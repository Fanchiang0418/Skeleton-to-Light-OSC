# Skeleton-to-Light-OSC
將人體骨架資料轉換為燈光控制數據的即時系統，透過 OSC 串接 TouchDesigner。

1.解析字串 (Analysis.py) (track_2.py) 

- 骨架資料形狀: (2657, 33, 3)

  - 2657 → 影片總共有 2657 幀 (frames)

  - 33 → 每幀有 33 個關節點 (joints)

  - 3 → 每個關節有三個座標值 (x, y, z)

  - 把原本一堆字串格式 (‘0.5’,‘0.1’,‘-2.2’)，整理成一個乾淨的骨架資料陣列

2.骨架數值 > OSC > Touchdesigner (pose_viewer_osc.py) (track_2.py) (Test1.toe)

3.數值說明 : 
- intensity
