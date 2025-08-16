import numpy as np
from scipy.ndimage import label, binary_closing,find_objects,median_filter,grey_closing


# --- 1. 加载你的高质量PoCA密度图 ---
nx,ny,nz=100,100,20
poca_map = np.fromfile("PoCA_Density.img", dtype=np.float64).reshape(nx, ny, nz)
improved_initial_map = poca_map.copy()

improved_initial_map = median_filter(improved_initial_map, size=(3, 3, 3))
improved_initial_map = grey_closing(improved_initial_map, size=(3, 3, 3))



print("improved initial map")
improved_initial_map.tofile("improved_initial_map.img")
print(f"improved initial map max value:{np.max(improved_initial_map)}")

diff = np.linalg.norm(improved_initial_map-poca_map)
print(f"difference:{diff},relative difference:{diff/np.linalg.norm(poca_map)}")
