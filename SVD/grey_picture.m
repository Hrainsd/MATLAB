% 单张灰色图片SVD压缩
ratio = 0.9;
pho_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\赫本.jpg";
save_add = 'C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\compressed_赫本.jpg';
pic = double(imread(pho_add));
[real_ratio,compressed_pic] = oursvd(pic, ratio);
imwrite(uint8(compressed_pic), save_add);