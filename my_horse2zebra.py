from PIL import Image
import numpy as np
import os

def readDirFile(path):
    fileLists = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if ".jpg" in filename.lower():  # check whether the file's DICOM
                fileLists.append(os.path.join(dirName,filename))
    return fileLists
def img2numpy(path):
    file_list = readDirFile(path)
    buf=[]
    for path in file_list:
        temp_im = Image.open(path).convert("RGB")#恒定为三通道 因为trainB中有异常图线需要做次处理
        buf.append(np.array(temp_im))
    buf=np.stack(buf,axis=0)
    # axis指定新轴在结果尺寸中的索引 当axis=0 即batch维度
    # 总元素个数为len(buf)*buf.shape[0]*buf.shape[1]...buf.shape[n]
    # 堆叠后shape为 [... len(buf) ...] len(buf)在shape中的索引值(位置) 即axis指定的值
    # 这些计算完成后,按照从后往前的顺序开始堆叠 遇到len(buf)所在维度 即axis指定维度时 该维度所有元素是list中各成员对应位置(其余维度存在且确定)的单一元素贡献出来的集合
    return buf

def load_horse2zebra(headpath,get_new=True,detype=np.uint8):
    """
    get_new=False只有在已经存在horse2zebra.npz时有效 就不通过os.walk() 判定了 超出了复杂度
    normalization会出现新的实例 不建议放在此处
    """
    if get_new == True:
        testA = img2numpy(headpath+"testA/")
        testB = img2numpy(headpath+"testB/")
        trainA = img2numpy(headpath+"trainA/")
        trainB = img2numpy(headpath+"trainB/")
        np.savez(headpath+"horse2zebra.npz",k1=testA,k2=testB,k3=trainA,k4=trainB)
    else:
        npzfile=np.load(headpath+'horse2zebra.npz') 
        testA = npzfile['k1']
        testB = npzfile['k2']
        trainA = npzfile['k3']
        trainB = npzfile['k4']
    return (testA.astype(detype),testB.astype(detype)),(trainA.astype(detype),trainB.astype(detype))
if __name__ == "__main__":
    (testA,testB),(trainA,trainB)=load_horse2zebra("./datasets/horse2zebra/",get_new=True,detype=np.uint8)
    print(testA.shape,testA.dtype)
    print(testB.shape,testB.dtype)
    print(trainA.shape,trainA.dtype)
    print(trainB.shape,trainB.dtype)

