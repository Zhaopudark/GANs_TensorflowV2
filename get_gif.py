import imageio
import os
import sys

def readDirFile(path):
    """
    从给定目录中获取dicom文件（.dcm）的详细路径，参数为指定目录名，返回路径组成的列表
    """
    fileLists = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                fileLists.append(os.path.join(dirName,filename))
    return fileLists

def create_gif(source, name, duration):
	"""
     生成gif的函数，原始图片仅支持png
     source: 为png图片列表（排好序）
     name ：生成的文件名称
     duration: 每张图片之间的时间间隔
	"""
	frames = []     # 读入缓冲区
	for img in source:
		frames.append(imageio.imread(img))#工作目录已经进入文件夹 所以每次都可将所有图片放入缓冲区
	imageio.mimsave(name, frames, 'GIF', duration=duration) #制图
	print("处理完成")

os.chdir("./DCGAN_WGP_C/") #将当前目录调整到图片文件夹中
pic_list = readDirFile("./")
# os.listdir()#将所有图片名记录
gif_name = "C_WGPDC_gan_result.gif" # 生成gif文件的名称
duration_time = 0.1 #这里似乎有最短间隔 再小就没有用了 和0.1 0.01 间隔似乎是一样的
# 生成gif
create_gif(pic_list, gif_name, duration_time)