import cv2
import numpy as np

from tqdm import tqdm

# 计算图像的梯度
def compute_smoothness_gradient(img):
    # 计算图像的拉普拉斯算子
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian

# 读取两帧图像
frame1 = cv2.imread('figs/Pig_frame1.jpg', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('figs/Pig_frame2.jpg', cv2.IMREAD_GRAYSCALE)
height, width = frame1.shape
print('frame1\'s shape:', frame1.shape)
print('frame2\'s shape:', frame2.shape)

# 放缩图像
scale_percent = 25  # 设定缩放比例
width = int(frame1.shape[1] * scale_percent / 100)
height = int(frame1.shape[0] * scale_percent / 100)
dim = (width, height)
frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
print('after resizing, frame1\'s shape:', frame1.shape)
print('after resizing, frame2\'s shape:', frame2.shape)

# 设置 Horn-Schunck 光流参数
alpha = 1.0  # 光流平滑度
iterations = 10000  # 迭代次数
epsilon = 1e-8  # 停止条件
lam = 100

# 计算梯度
Ix = cv2.Sobel(frame1, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(frame1, cv2.CV_64F, 0, 1, ksize=3)
It = frame2 - frame1

cv2.imwrite('result/Ix.jpg', Ix)
cv2.imwrite('result/Iy.jpg', Iy)
cv2.imwrite('result/It.jpg', It)
# cv2.imshow('Gradient of x axis', Ix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 初始化 U、V 光流估计
U = np.zeros_like(frame1, dtype=np.float64)
V = np.zeros_like(frame1, dtype=np.float64)

# 迭代求解 Horn-Schunck 光流
pbar = tqdm(range(iterations), desc='calculating flow')
for e in pbar:
    temp = np.array(
        [
            [1/12, 1/6, 1/12],
            [1/6, 0, 1/6],
            [1/12, 1/6, 1/12]
        ]
    )
    temp = np.array(
        [
            [0, 1/2, 0],
            [0, 0, 1/2],
            [0, 0, 0]
        ]
    )
    # temp = np.array(
    #     [
    #         [1/24, 1/24, 1/24, 1/24, 1/24],
    #         [1/24, 1/24, 1/24, 1/24, 1/24],
    #         [1/24, 1/24, 0, 1/24, 1/24],
    #         [1/24, 1/24, 1/24, 1/24, 1/24],
    #         [1/24, 1/24, 1/24, 1/24, 1/24],
    #     ]
    # )
    u_avg = cv2.filter2D(U, -1, temp)
    v_avg = cv2.filter2D(V, -1, temp)

    new_U = u_avg - (Ix * u_avg + Iy * v_avg + It) * Ix / (lam + Ix ** 2 + Iy ** 2)
    new_V = v_avg - (Ix * u_avg + Iy * v_avg + It) * Iy / (lam + Ix ** 2 + Iy ** 2)

    dU = new_U - U
    dV = new_V - V

    # 检查迭代是否收敛
    pbar.set_postfix(MSE=np.sum(np.square(dU)) + np.sum(np.square(dV)), refresh=True)
    if np.sum(np.square(dU)) + np.sum(np.square(dV)) < epsilon:
        break

    U = new_U
    V = new_V

# 可视化光流
# 创建一个空白图像，用于绘制箭头
flow_img = frame1.copy()

# 绘制箭头
step = 10  # 控制箭头密度
for y in range(0, flow_img.shape[0], step):
    for x in range(0, flow_img.shape[1], step):
        cv2.arrowedLine(flow_img, (x, y), (int(x + U[y, x]), int(y + V[y, x])), (0, 255, 0), 1)

# 显示结果
cv2.imwrite(f'result/flow_{lam}_s_24.jpg', flow_img)
# cv2.imshow('Optical Flow', flow_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 全局运动预测
bar = tqdm(total=int(height*width), desc='motion prediction')
predicted_frame = frame1.copy()
for y in range(0, height):
    for x in range(0, width):
        u = U[y, x]
        v = V[y, x]
        new_x = int(x + u)
        new_y = int(y + v)
        if 0 <= new_x < width and 0 <= new_y < height:
            predicted_frame[new_y, new_x] = frame1[y, x]
        bar.update(1)
bar.close()

# 显示结果
cv2.imwrite(f'result/predicte_{lam}_s_24.jpg', predicted_frame)
# cv2.imshow('Predicted Frame', predicted_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

err = np.sum((predicted_frame - frame2) ** 2) / height / width
print('MSE:', err)
