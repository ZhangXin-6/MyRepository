clc;
clear;
close all;
% disp = imread('D:\zhangxin\DSP-Result\IGEV-OMA\OMA042_001_002_LEFT_RGB.tif');
% disp = double(disp);
% mindisp = min(disp);
% disp = disp - mindisp;
% img = imread('D:\zhangxin\dataset\US3D\Track2-RGB-3\OMA042_001_002_LEFT_RGB.tif');

% 读取视差图（灰度图）
disparity_map = imread('C:\Users\ZX\Documents\笔记\图片\YD_left_495.tif');
% disparity_map = imread('D:\zhangxin\dataset\US3D\Track2-RGB-3\OMA042_001_002_LEFT_AGL.tif'); % 视差图路径
disparity_map = im2double(-disparity_map); % 归一化到 [0,1]

% 读取彩色图像
rgb_image = imread('C:\Users\ZX\Documents\笔记\图片\2.png'); % 颜色图路径
rgb_image = im2double(rgb_image); % 转换为double

% 获取图像大小
[rows, cols] = size(disparity_map); 
[x, y] = meshgrid(1:cols, 1:rows); 

% 将视差值用于深度（可乘以系数调整高度比例）
z = disparity_map; % 乘以 50 让高度明显（可调整）

% 旋转视差图使得它的方向与 MATLAB 的绘图方向匹配
z = flipud(z);
rgb_image = flipud(rgb_image);

% 绘制3D网格并映射颜色
figure;
surf(x, y, z, rgb_image, 'FaceColor', 'texturemap');

% 设置渲染参数
shading interp;  % 平滑颜色过渡
axis tight;      % 让坐标轴适应数据范围
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Depth (Disparity-based Height)');
title('3D Disparity Map with RGB Texture');

grid on;
view(3); % 3D 视角