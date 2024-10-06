% 单张彩色图片SVD压缩
ratio = 0.9;
pho_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\千与千寻.jpg";
save_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\compressed_千与千寻.jpg";
pic = double(imread(pho_add));
R=pic(:,:,1); % red   
G=pic(:,:,2); % green
B=pic(:,:,3); % blue
[real_ratio1,compressed_r] = oursvd(R, ratio);  
[real_ratio2,compressed_g] = oursvd(G, ratio); 
[real_ratio3,compressed_b] = oursvd(B, ratio); 
compress_pic=cat(3,compressed_r,compressed_g,compressed_b);
imwrite(uint8(compress_pic), save_add);