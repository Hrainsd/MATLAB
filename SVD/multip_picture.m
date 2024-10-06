% 文件夹多张(图片处理)
% 彩色
fdr = 'C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\压缩文件夹内的所有图片\葫芦娃七兄弟';  
dt=dir(fullfile(folder_name, '*.jpg'));  
cell_name={dt.name};
n = length(cell_name);
ratio = 0.9;
for i = 1:n
    pto_name = cell_name(i);
    name = pto_name{1};  
    pto_add = fullfile(fdr, name); 
    save_add = fullfile(fdr,['compressed_',name]);
    pic = double(imread(pto_add));
    R=pic(:,:,1); % red   
    G=pic(:,:,2); % green
    B=pic(:,:,3); % blue
    [real_ratio1,compressed_r] = oursvd(R, ratio);  
    [real_ratio2,compressed_g] = oursvd(G, ratio); 
    [real_ratio3,compressed_b] = oursvd(B, ratio); 
    compress_pic=cat(3,compressed_r,compressed_g,compressed_b);
    imwrite(uint8(compress_pic), save_add);
end
% 灰色
fdr = 'C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\压缩文件夹内的所有图片\葫芦娃七兄弟';  
dt=dir(fullfile(folder_name, '*.jpg'));  
cell_name={dt.name};
n = length(cell_name);
ratio = 0.9;
for i = 1:n
    pto_name = cell_name(i);
    name = pto_name{1};  
    pto_add = fullfile(fdr, name); 
    save_add = fullfile(fdr,['compressed_',name]);
    pic = double(imread(pto_add));
    [real_ratio,compressed_pic] = oursvd(pic, ratio);
    imwrite(uint8(compress_pic), save_add);
end