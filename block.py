import cv2
import numpy as np
from tqdm import tqdm

# 定义块大小和搜索范围
block_size = 8
search_range = 16

# 读取两帧图像
frame1 = cv2.imread('figs/Pig_frame1.jpg', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('figs/Pig_frame2.jpg', cv2.IMREAD_GRAYSCALE)
print('frame1\'s shape:', frame1.shape)
print('frame2\'s shape:', frame2.shape)

# 放缩图像
scale_percent = 25  # 设定缩放比例
width = int(frame1.shape[1] * scale_percent / 100 / block_size) * block_size
height = int(frame1.shape[0] * scale_percent / 100 / block_size) * block_size
dim = (width, height)
frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
print('after resizing, frame1\'s shape:', frame1.shape)
print('after resizing, frame2\'s shape:', frame2.shape)

# 获取图像尺寸
height, width = frame1.shape

# 初始化存储运动向量的数组
motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.float32)

# 对每个块进行运动估计
bar = tqdm(total=int(height*width/(block_size**2)), desc='motion estimation')
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        # 获取当前块
        block1 = frame1[y:y+block_size, x:x+block_size]
        
        # 初始化最小均方差和对应的运动向量
        min_mse = float('inf')
        best_motion_vector = (0, 0)
        
        # 在搜索范围内寻找最匹配的块
        for dy in range(-search_range, search_range+1):
            for dx in range(-search_range, search_range+1):
                # 确定搜索窗口边界
                x_start = max(x + dx, 0)
                x_end = min(x + dx + block_size, width)
                y_start = max(y + dy, 0)
                y_end = min(y + dy + block_size, height)
                
                # 获取当前搜索窗口
                block2 = frame2[y_start:y_end, x_start:x_end]
                
                # 如果搜索窗口尺寸与当前块不同，则跳过
                if block1.shape != block2.shape:
                    continue
                
                # 计算均方差
                mse = np.mean((block1 - block2) ** 2)
                
                # 更新最小均方差和对应的运动向量
                if mse < min_mse:
                    min_mse = mse
                    best_motion_vector = (dx, dy)
        
        # 存储运动向量
        motion_vectors[y // block_size, x // block_size] = best_motion_vector
        bar.update(1)
bar.close()

# 绘制运动向量
new_frame1 = frame1.copy()
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        dx, dy = motion_vectors[y // block_size, x // block_size]
        cv2.arrowedLine(new_frame1, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), 1)

# 显示结果
cv2.imwrite('result/block_flow.jpg', new_frame1)
cv2.imshow('Optical Flow', new_frame1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 全局运动预测
bar = tqdm(total=int(height*width), desc='motion prediction')
predicted_frame = frame1.copy()
for y in range(0, height):
    for x in range(0, width):
        u = motion_vectors[y // block_size, x // block_size, 0]
        v = motion_vectors[y // block_size, x // block_size, 1]
        new_x = int(x + u)
        new_y = int(y + v)
        if 0 <= new_x < width and 0 <= new_y < height:
            predicted_frame[new_y, new_x] = frame1[y, x]
        bar.update(1)
bar.close()

# 显示结果
cv2.imwrite('result/predict.jpg', predicted_frame)
cv2.imshow('Predicted Frame', predicted_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

err = np.sum((predicted_frame - frame2) ** 2) / height / width
print('MSE:', err)