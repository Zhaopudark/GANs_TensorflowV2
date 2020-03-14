from __future__ import absolute_import, division, print_function, unicode_literals
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Dropout(layers.Layer):
    def __init__(self,in_shape,dropout_rate=0.3):
        super(Dropout,self).__init__()
        self.in_shape = in_shape
        self.out_shape = self.in_shape
        self.dropout_rate = dropout_rate
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.nn.dropout(x,self.dropout_rate)
            return y
        else:
            y = x
            return y
class LeakyReLU(layers.Layer):
    def __init__(self,in_shape):
        super(LeakyReLU,self).__init__()
        self.in_shape = in_shape
        self.out_shape = self.in_shape
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.nn.leaky_relu(x)
            return y
        else:
            y = tf.nn.leaky_relu(x)
            return y
class Dense(layers.Layer):
    def __init__(self, input_dim, units,use_bias=True):
        super(Dense,self).__init__()
        initializer = tf.initializers.glorot_uniform()
        # initializer = tf.initializers.glorot_normal()
        self.w = tf.Variable(initializer((input_dim,units)),dtype=tf.float32,trainable=True)
        self.b = tf.Variable(tf.zeros((1,units)),dtype=tf.float32,trainable=use_bias)#节点的偏置也是行向量 才可以正常计算 即对堆叠的batch 都是加载单个batch内
        self.out_dim = units
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.matmul(x,self.w)+self.b
            return y
        else:
            y = tf.matmul(x,self.w)+self.b
            return y 
class BatchNormalization(layers.Layer):
    def __init__(self,in_shape):
        super(BatchNormalization,self).__init__()
        """
        in_shape 不考虑batch维度
        """
        self.beta = tf.Variable(tf.zeros((1)),trainable=True)
        self.gamma = tf.Variable(tf.ones((1)),trainable=True)
        self.global_u = tf.Variable(tf.zeros(in_shape),dtype=tf.float32,trainable=False)#参与测试不参与训练
        self.global_sigma2=tf.Variable(tf.ones(in_shape),dtype=tf.float32,trainable=False)#参与测试不参与训练
        self.in_shape = in_shape
        self.out_shape = self.in_shape

    @tf.function
    def call(self,x,training=True):
        """
        x = tf.constant([[1., 1.], [2., 2.]]) [y,x]
        tf.reduce_mean(x)  # 1.5
        tf.reduce_mean(x, 0)  # [1.5, 1.5]  固定每个x 对y维所有元素求均值
        tf.reduce_mean(x, 1)  # [1.,  2.]  固定每个y 对x为所有元素求均值
        tf.reduce_mean(x,axis) axis=0 参数 表示 对于固定的非0维  即一个确定的1 2 3 。。。维度 对应的所有0维元素求均值 然后删掉该维度 该维度维数是1 删掉整体降维但是不影响元素个数 如果删除后出现列向量 则转为行向量 
        """
        #imput 是x shape=[batch_size,x,x,x]
        if training== True:
            u = tf.reduce_mean(x,axis=0)
            sigma2 = tf.reduce_mean(tf.math.square(x-u),axis=0)

            self.global_u.assign(self.global_u*0.99+0.01*u)
            self.global_sigma2.assign(self.global_sigma2*0.99+0.01*sigma2)

            x_hat = (x-u)/tf.math.sqrt(sigma2+0.001) #计算出均值后归一化 确实应当保证输入维度不变
            y = self.gamma*x_hat+self.beta
            return y
        else:#training== False
            x_hat = (x-self.global_u)/tf.math.sqrt(self.global_sigma2+0.001) #计算出均值后归一化 确实应当保证输入维度不变
            y = self.gamma*x_hat+self.beta
            return y

class Conv2DTranspose(layers.Layer):
    def __init__(self,in_shape,out_depth,kernel_size,strides=[1,1],pandding_way="SAME",use_bias=False):
        super(Conv2DTranspose,self).__init__()
        if len(strides)<=2:
            self.strides = [1]+strides+[1] #满足[1,h,w,1]的形式  但是按照官网的意思 应该是0xx0 才对 不知道具体怎么样 但是知道的是 这里正常是前后补1 不改变参数量的升维
        else:
            pass
        self.pandding_way = pandding_way
        self.out_depth = out_depth
        initializer = tf.initializers.glorot_uniform()
        #反卷积的全举证以[height, width, output_channels, in_channels]方式定义 依据了tf.nn.conv2d_transpose的要求
        w_shape = kernel_size+[self.out_depth]+[in_shape[-1]]
        self.w = tf.Variable(initializer(w_shape),dtype=tf.float32,trainable=True)
        P = [None,None]
        if self.pandding_way =="SAME":
            P[0]= kernel_size[0]//2
            P[1]= kernel_size[1]//2
            out_shape= [self.strides[1]*in_shape[0],self.strides[2]*in_shape[1]]
        elif self.pandding_way == "VALID":
            P = [0,0]
            out_shape= [self.strides[1]*in_shape[0]+max(kernel_size[0]- self.strides[1],0),self.strides[2]*in_shape[1]+max(kernel_size[1]-self.strides[2],0)]
            # 该公式照抄了源码 原理也是依据下述推导 对向下取整产生多值时取了一个合理定制 stride=2时不可避免反卷积出现维度歧义
            # 可见反卷积的定义不应该从卷积反过来看 应当有个更好的定义拒绝歧义
        else:
            raise ValueError
        self.b_shape = out_shape+[self.out_depth]
        self.b = tf.Variable(tf.zeros(self.b_shape),dtype=tf.float32,trainable=use_bias)
        self.in_shape = in_shape
        self.out_shape = self.b_shape
        """
        非正方形卷积核 特征图 也是适用的 为了方便 假定是正方形 只阐述横向方向
        卷积 [B,H1,W1,C1] 到 [B,H2=?,W2=?,C2]的变化 给定卷积核的大小Wk 
        为了确定卷积核参数个数 需要知道 C2 Hk Wk C1 参数个数=C2*Hk*Wk*C1
        为了确定偏置参数个数 需要知道 W2 参数个数=H2*W2*C2 有公式如下
        W2 = [(W1-Wk+2*P)/S]+1 卷积的公式是始终成立的 []表示向下取整

        反卷积 在表面上 是将上述过程倒过来 使用相同大小的卷积核(卷积矩阵相同 但是其组数和层数不同 依据前后通道数的变化而来)和步长s 实现从[B,H2,W2,C2] 到 [B,H1',W1',C1']的变换 原则是使用相同大小的卷积核(参数内容必然是改变的,卷积核的组数 和 卷积矩阵个数必然是相反的 他们对应前后不同的通道)和步长s
        原则是使用相同大小的卷积核(参数内容必然是改变的)和步长S！！！ 所以才叫反卷积
        原则是使用相同大小的卷积核(参数内容必然是改变的)和步长S！！！ 所以需要给定卷积核
        原则是使用相同大小的卷积核(参数内容必然是改变的)和步长S！！！ 所以采用相同大小的另外的卷积核 和步长S 是可以从[H1',W1']变换到
        实际上  是对[B,H2,W2,C2]依据给定的 反卷积核  反卷积padding 进行正常的卷积操作 只是padding的方式有所变化 有可能拉散原数据 中间补零(依据padding方式) 实现到[B,H1',W1',C1']的变换 一般情况都是将单个通道的图像放大了

        即如果我们知道了W2 Wk padding strides 可以推算出W1 只要找到适合上述的W1即可 
        问题在于 对于向下取整[] W1的取值可能是不唯一的 所以存在多解 
        
        那 若是我们再给定了原输入维度W1(即期望反卷积输出维度) 只要不予上面的多解冲突 即是可行的 

        按照tf.keras.layers.Conv2DTranspose 的输入要求 Conv2DTranspose虽然明面上不需要知道输入维度 但是是自动关联上一层的输出了  第一层时需要指定 故还是需要知道(H2,W2) C2
        已知 (H2,W2) C2 C1' [Hk,Wk] S=[S_h,S_w] 和padding方式 
        依据    padding="SAME" P_w=Wk//2,P_h=Hk//2   
                padding="VALID" P_w=P_h=0
        约束为
                W2 = [(W1'-Wk+2*P_w)/S_w]+1
                H2 = [(H1'-Hk+2*P_h)/S_h]+1
        卷积核参数个数为 C1'*Hk*Wk*C2
        偏置参数个数为   H1'*W1'*C1'

        tensorflow 有意思的是 tf.nn.conv2d_transpose要求的输入中 有input 卷积核 output_shape,strides,padding='SAME' 即
        可以知道 (H2,W2) C2  Hk Wk     H1' W1' C1' S=[S_h,S_w] padding  那只需要验证等式是否成立就好了

        大胆猜想 tf.keras.layers.Conv2DTranspose对上述的约束采取了极端形式 直接给定了因为取值而多值的H1' W1'
        而tf.nn.conv2d_transpose则要求用户给定H1' W1' 防止有歧义

        我基于tf.nn.conv2d_transpose构建自己的layers.Conv2DTranspose时还是需要计算并敲定W1' H1'的 不然就无法使用偏置了 也无法完成前向过程
        按照tf.keras.layers.Conv2DTranspose源码中 对于"SAME"的Padding方式 反卷积直接采用W1'=W2*S_w 的方式求值 我这里也就不再去做多值的计算了  
        """
    @tf.function
    def call(self,x):
        convtranspose_out = tf.nn.conv2d_transpose(input=x,filters=self.w,output_shape=[x.shape[0]]+self.b_shape,strides=self.strides, padding=self.pandding_way)
        l_out = convtranspose_out+self.b
        return l_out
class Conv2D(layers.Layer):
    def __init__(self,input_shape,out_depth,filter_size,strides,use_bias=True,pandding_way="SAME"):
        super(Conv2D,self).__init__()
        if len(strides)<=2:
            self.strides = [1]+strides+[1] #满足[1,h,w,1]的形式
        else:
            pass
        self.pandding_way = pandding_way
        """
        非正方形卷积核 特征图 也是适用的 为了方便 假定是正方形 只阐述横向方向
        input_shape 是图片大小和通道数
        kernel_size 是卷积核大小
        step是步长stride
        kernel_initializer='glorot_uniform', bias_initializer='zeros'

        对一个batch中的特定一个输入
        输入特征图为 [W1,H1,D1] 三个参数 即特征图宽W1 高H1 和深度-通道D1
        
        输出特征图为 [W2,H2,D2] 三个参数 即特征图宽W2 高H2 和深度-通道D2

        卷积核和偏置维系了输入输出特征图的关系

        卷积核有 四个参数 [D2,Wk,Hk,D1]  对于输入特征图中的一个通道 对于一组卷积核中的一个
        即 W1 H1 D1 对应 Wk Hk D1

        输入特征图的一个通道只会有唯一的一个卷积矩阵和其卷积  
        一组卷积核中的一个卷积矩阵只与对应的一个输入特征通道卷积 

        所以一组卷积核中的一个卷积矩阵 和  输入特征中的一个通道 是一一对应的 不会和其他的有关联

        卷积核的组数 类似于batch但却不是  由输出特征图通道数决定

        即多少组卷积核 就有多少个输出特征通道 

        所以  对于一个卷积操作 是将每一组卷积核中的D1个卷积矩阵[Wk,Hk] 和输入特征图的D1个通道[W1,H1] 一对一 分别卷积 不互相干扰 得到D1数量的卷积输出[W2,H2] 将它们相加 
        得到一个[W2,H2]矩阵 构成不加偏置的 输出特征图的一个通道 加上 一个[W2,H2] 的偏置矩阵 得到带偏置的输出特征图的一个通道

        对D2组卷积核做同样的操作  就得到了 D2个 [W2,h2]的输出特征图 D2就是输出特征图通道数

        W1 H1 D1 D2 都是指定的 下面需要计算W2 H2的关系 对于padding 理解成先改变W1 H1 再进行上述的操作即可
        辅之 padding的大小P  卷积步长S 指定"SAME"时 padding 在输入特征图的每个通道的周围补多层0值 保证如果是步长为1 卷积前后特征图的大小不变 通道可能改变 P=Wk/2
                                    指定"VALID"时 不padding P=0
        卷积前后 特征图的宽的变化规律如下 
        W2 = [(W1-Wk+2*P)/S]+1 这个公式的理解是  最左上角一定是卷积第一次运算 然后剩余部分可以让卷积核移动几次呢 然后相加即可 [.]向下取整 因为当不能整除时 鉴于第一个位置的存在 就不够距离放下下一个卷积核了
        W2 = {(W1-Wk+2*P+1)/S} 和上面的公式等价 理解为 计算出卷积核的最上行或者最左列可以出现的位置个数 在步长s的区间上分配 整除则刚好 不整除则需要进一法补全 因为确实可以每S个区间放下一个 且等间距
        //双斜杠才是整除取整向下

        已知 (H1,W1) C1  C2 [Hk,Wk] S=[S_h,S_w] 和padding方式 
        依据    padding="SAME" P_w=Wk//2,P_h=Hk//2   
                padding="VALID" P_w=P_h=0
        约束为 
                W2 = [(W1-Wk+2*P_w)/S_w]+1
                H2 = [(H1-Hk+2*P_h)/S_h]+1
        卷积核参数个数为 C2*Hk*Wk*C1
        偏置参数个数为   H2*W2*C2
        """
        P = [None,None]
        if self.pandding_way == "SAME":
            P[0] = filter_size[0]//2
            P[1] = filter_size[1]//2
        elif self.pandding_way=="VALID":
            P = [0,0]
        else:
            raise ValueError
        out_shape = [(input_shape[i]+2*P[i]-filter_size[i])//self.strides[i+1]+1 for i in range(2)]

        initializer = tf.initializers.glorot_uniform()
        w1_shape = filter_size+[input_shape[-1]]+[out_depth]
        self.w = tf.Variable(initializer(w1_shape),dtype=tf.float32,trainable=True)

        b_shape = out_shape+[out_depth]
        self.b = tf.Variable(tf.zeros(b_shape),dtype=tf.float32,trainable=use_bias)
        self.in_shape = input_shape
        self.out_shape = b_shape
    @tf.function
    def call(self,x):
        conv_out = tf.nn.conv2d(input=x,filters=self.w,strides=self.strides,padding=self.pandding_way,data_format='NHWC',dilations=None,name=None) #dilations是空洞卷积的一个系数 相当于对卷积核做上采样同时部分置零  这里不进行空洞卷积
        l_out = conv_out+self.b
        return l_out

if __name__ == "__main__":
    (train_images,train_labels),(_, _) = my_mnist.load_data(get_new=False,
                                                        normalization=False,
                                                        one_hot=True,
                                                        detype=np.float32)
    train_images = (train_images.astype('float32')-127.5)/127.5
    train_labels = (train_labels.astype('float32')-0.5)/0.5                                                    
    train_images = train_images.reshape(train_images.shape[0], 28, 28,1)
    print(train_labels[0])
    plt.imshow(train_images[0, :, :,0], cmap='gray')
    plt.show()

    x = tf.random.normal(shape=(64,784))
    a = Dense(28*28,128)
    print(a(x))
    print(len(a.trainable_variables))
    y = train_images[0:1, :, :,0:1]
    C1 = Conv2D([28,28,1],2,[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
    print(len(C1.trainable_variables)) #卷积操作的输入必须满足N(B) H W C 
    fielt_out = C1(y)
    plt.imshow(fielt_out[0, :, :,0], cmap='gray')
    plt.show()
    plt.imshow(fielt_out[0, :, :,1], cmap='gray')
    plt.show()
    C2 = Conv2D([28,28,1],2,[5,5],strides=[2,2],use_bias=False,pandding_way="SAME")
    print(len(C2.trainable_variables))
    C3 = Conv2DTranspose([28,28,1],2,[5,5],strides=[1,1],pandding_way="SAME",use_bias=False)
    print(len(C3.trainable_variables))
    filter_out = C3(y)
    plt.imshow(filter_out[0, :, :,0], cmap='gray')
    plt.show()
    plt.imshow(filter_out[0, :, :,1], cmap='gray')
    plt.show()
    C4 = Conv2DTranspose([28,28,1],2,[5,5],strides=[1,1],pandding_way="SAME",use_bias=True)
    print(len(C4.trainable_variables))

    B1 = BatchNormalization(in_shape=[28,28,1])
    x = train_images[0:2, :, :,:]
    print(B1(x,training=True))
    print(B1(x,training=False))
    print(B1.trainable_variables)
    print(len(B1.trainable_variables))