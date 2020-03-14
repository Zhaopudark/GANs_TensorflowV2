from __future__ import absolute_import, division, print_function, unicode_literals
"""
基于tensorflow 高阶API
WGAN  Wasserstein GAN + DCGAN +Conditional
损失函数: WGAN W 距离loss  考察真假样本的分布差异 判别器需要满足Lipschitz-1 Lipschitz-K 约束 所以不能加sigmoid约束范围
        通过对输入的求导 得到关于输入的导数 即权值W的函数 作为正则项  对其值进行直接约束 从而 满足Lipschitz-1 Lipschitz-K 约束
网络结构: 多层的卷积形式 判别器在卷积层后的全连接层concat One_Hot条件  而 生成器在开头concat One_Hot 条件
数据形式: 带卷积层 数据映射到-1 1 区间
生成器: tanh 映射到-1 1 之间 迎合数据格式 
判别器: 最后一层 没有sigmoid 没有relu 直接是matmul运算结果  迎合loss公式的约束 在全域内寻找满足Lipschitz-1 Lipschitz-K 约束的函数
初始化: xavier初始化  即考虑输入输出维度的 glorot uniform
训练： 判别器5次 生成器1次
"""
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import my_layers
from tensorflow.keras import layers
import time
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

    
def sample_z(shape):
    # return tf.random.normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32)
    return tf.random.uniform(shape,minval=0,maxval=1.0,dtype=tf.float32)
class Discriminator(tf.keras.Model):
    def __init__(self,in_shape,label_dim):
        super(Discriminator,self).__init__()
        """
        反卷积和dense层采用偏置 各自2参数 
        2+2+2=6 一共六个参数个数(指独立大参数self.w self.b的个数)
        """
        self.Conv2d_1 = my_layers.Conv2D(input_shape=in_shape,out_depth=64,filter_size=[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
        self.LeakyReLU_1 = my_layers.LeakyReLU(in_shape=self.Conv2d_1.out_shape)
        self.DropOut_1 = my_layers.Dropout(in_shape=self.LeakyReLU_1.out_shape,dropout_rate=0.3)
        
        self.Conv2d_2 = my_layers.Conv2D(input_shape=self.DropOut_1.out_shape,out_depth=128,filter_size=[5,5],strides=[2,2],use_bias=True,pandding_way="SAME")
        self.LeakyReLU_2 = my_layers.LeakyReLU(in_shape=self.Conv2d_2.out_shape)
        self.DropOut_2= my_layers.Dropout(in_shape=self.LeakyReLU_2.out_shape,dropout_rate=0.3)
        next_shape = 1
        for i in self.DropOut_2.out_shape:
            next_shape *= i 
        # self.Dense = my_layers.Dense(next_shape,units=1)
        self.Dense = my_layers.Dense(next_shape,units=100)
        self.Dense_conditional_1 = my_layers.Dense(self.Dense.out_dim+label_dim,units=50)
        self.Dense_conditional_2 = my_layers.Dense(self.Dense_conditional_1.out_dim,units=1)
    @tf.function
    def call(self,x,y,training=True):
        conv2_l1 = self.Conv2d_1(x)
        leakey_relu_l1 = self.LeakyReLU_1(conv2_l1,training)
        dropout_l1 = self.DropOut_1(leakey_relu_l1,training)
        conv2_l2 = self.Conv2d_2(dropout_l1)
        leakey_relu_l2 = self.LeakyReLU_2(conv2_l2,training)
        dropout_l2 = self.DropOut_2(leakey_relu_l2,training)
        dense_l3 =  self.Dense(tf.reshape(dropout_l2,[dropout_l2.shape[0],-1]),training)
        # l3_out = tf.nn.sigmoid(dense_l3)
        l3_out = tf.nn.tanh(dense_l3)
        l4_out = tf.nn.leaky_relu(self.Dense_conditional_1(tf.concat([l3_out,y],axis=1),training))
        l5_out = self.Dense_conditional_2(l4_out,training)
        return l5_out
    @tf.function
    def clip_op(self):
        for item in self.trainable_variables:
            temp = item
            temp.assign(tf.clip_by_value(temp,clip_value_min=-0.01, clip_value_max=0.01))#clip 操作返回tensor assign赋值

d = Discriminator(in_shape=[28,28,1],label_dim=10)
x = train_images[0:128, :, :,:]
y = train_labels[0:128,:]
print(d(x,y,training=False))#行向量统一输入  而batch是行向量在列方向堆叠后的矩阵 
# print(d(x,training=True))#行向量统一输入  而batch是行向量在列方向堆叠后的矩阵 
print(len(d.trainable_variables))

class Generator(tf.keras.Model):
    def __init__(self,in_dim):
        super(Generator,self).__init__()
        """
        bn层两个参数 
        反卷积和dense层不采用偏置 各自只有一个参数 
        1+2+1+2+1+2+1=10 一共十个参数个数(指独立大参数self.w self.b的个数)
        """
        self.Dense_1 = my_layers.Dense(in_dim,7*7*256,use_bias=False)
        self.BacthNormalization_1 = my_layers.BatchNormalization(in_shape=self.Dense_1.out_dim)
        self.LeakyReLU_1 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_1.out_shape)
        
        self.Conv2dTranspose_2 = my_layers.Conv2DTranspose(in_shape=[7,7,256],out_depth=128,kernel_size=[5,5],strides=[1,1],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_2.out_shape == [7,7,128]
        self.BacthNormalization_2 = my_layers.BatchNormalization(in_shape=self.Conv2dTranspose_2.out_shape)
        self.LeakyReLU_2 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_2.out_shape)

        self.Conv2dTranspose_3 = my_layers.Conv2DTranspose(in_shape=self.LeakyReLU_2.out_shape,out_depth=64,kernel_size=[5,5],strides=[2,2],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_3.out_shape == [14,14,64]
        self.BacthNormalization_3 = my_layers.BatchNormalization(in_shape=self.Conv2dTranspose_3.out_shape)
        self.LeakyReLU_3 = my_layers.LeakyReLU(in_shape=self.BacthNormalization_3.out_shape)

        self.Conv2dTranspose_4 = my_layers.Conv2DTranspose(in_shape=self.LeakyReLU_3.out_shape,out_depth=1,kernel_size=[5,5],strides=[2,2],pandding_way="SAME",use_bias=False) 
        assert self.Conv2dTranspose_4.out_shape == [28,28,1]
    @tf.function
    def call(self,x,y,training=True):
        x = tf.concat([x,y],axis=1) #1维度相加 其他维度固定不变
        dense_l1 = self.Dense_1(x,training)
        #tf.print(dense_l1)
        bn_l1 = self.BacthNormalization_1(dense_l1,training)
        #tf.print(bn_l1) batch_normalization 在训练时 如果batch sizez是1 则会直接归零 因为会减去均值
        lr_l1 = self.LeakyReLU_1(bn_l1,training)
        #tf.print(lr_l1)
        conv2d_tr_l2 = self.Conv2dTranspose_2(tf.reshape(lr_l1,[-1,7,7,256]))
        bn_l2 = self.BacthNormalization_2(conv2d_tr_l2,training)
        lr_l2 = self.LeakyReLU_2(bn_l2,training)

        conv2d_tr_l3 = self.Conv2dTranspose_3(lr_l2)
        bn_l3 = self.BacthNormalization_3(conv2d_tr_l3,training)
        lr_l3 = self.LeakyReLU_3(bn_l3,training)

        conv2d_tr_l4 = self.Conv2dTranspose_4(lr_l3)
        l4_out = tf.nn.tanh(conv2d_tr_l4)
        return l4_out

g = Generator(100+10)
z = sample_z([2,100])
y = train_labels[0:2,:]
image = g(z,y,training=False)
for i in range(image.shape[0]):
    plt.imshow(tf.reshape(image[i],(28,28)), cmap='gray')
    plt.show()

# image = g(z,training=True)
# for i in range(image.shape[0]):
#     plt.imshow(tf.reshape(image[i],(28,28)), cmap='gray')
#     plt.show()
print(len(g.trainable_variables))
print(d(image,y,training=False))

def d_loss(real_output, fake_output):
    total_loss = -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)#用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
    return total_loss
def g_loss(fake_output):
    return -tf.reduce_mean(fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0,beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0,beta_2=0.9)

EPOCHS = 400
BATCH_SIZE = 128
z_dim = 100
num_examples_to_generate = 100

seed = sample_z([num_examples_to_generate, z_dim])
num_list = []
for i in range(10):
    num_list += [i]*10
print(num_list)
seed_lable= tf.one_hot(num_list,depth=10,on_value=1.0,off_value=-1.0,axis=-1,dtype=tf.float32) #axis理解成我们加入的深度10 在最终结果中的轴序号
print(seed_lable)
seed = [seed,seed_lable]

@tf.function
def D_train_step(images,labels):
    z = sample_z([images.shape[0], z_dim])
    with tf.GradientTape() as disc_tape:
        generated_images = g(z,labels,training=True)
        e = tf.random.uniform((images.shape[0],1,1,1),0.0,1.0) # [128,1]权重 无法与 [128,28,28,1]图片 相乘 需将权值变为[128,1,1,1]
        mid_images = e*images+(1-e)*generated_images
        with tf.GradientTape() as gradient_penalty:
            gradient_penalty.watch(mid_images)
            inner_loss = d(mid_images,labels,training=True)
        penalty = gradient_penalty.gradient(inner_loss,mid_images)
        penalty_norm = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1)# 这是我自己认为的  因为只有梯度大于1的才需要优化哇
                # penalty_norm = 10.0*tf.math.square(tf.norm(penalty,ord='euclidean')-1) 这是按照算法愿意
        real_output = d(images,labels,training=True)
        fake_output = d(generated_images,labels,training=True)
        # disc_loss = d_loss(real_output,fake_output)
        disc_loss = d_loss(real_output,fake_output)+tf.reduce_mean(penalty_norm)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    # d.clip_op() WGAN-GP 不需要clipping 
@tf.function
def G_train_step(images,labels):
    z = sample_z([images.shape[0], z_dim])
    with tf.GradientTape() as gen_tape:
        generated_images = g(z,labels,training=True)
        fake_output = d(generated_images,labels,training=True)
        gen_loss = g_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss,g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))

def train(train_images,train_labels,epochs):
    break_flag = 0
    index = list(range(train_images.shape[0]))
    np.random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
    for epoch in range(epochs):
        start = time.time()
        while True:
            for i in range(5):
                try:
                    x_real_bacth = next(images_batches)
                    y_label_bacth = next(labels_batches)
                    D_train_step(x_real_bacth,y_label_bacth)
                except StopIteration:
                    del images_batches
                    del labels_batches
                    np.random.shuffle(index)
                    train_images = train_images[index]
                    train_labels = train_labels[index]
                    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
                    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
                    break_flag = 1
                    break
            if break_flag == 0: # 判别器训练5次 然后进行一次生成器
                G_train_step(x_real_bacth,y_label_bacth)
            else:
                break_flag = 0
                break
        generate_and_save_images(g,epoch + 1,seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input[0],test_input[1],training=False)
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*127.5+127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('./DCGAN_WGP_C/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

print(time)
train(train_images,train_labels,EPOCHS)