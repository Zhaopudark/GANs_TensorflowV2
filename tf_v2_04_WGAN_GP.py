from __future__ import absolute_import, division, print_function, unicode_literals
"""
基于tensorflow 高阶API
WGAN-GP  Wasserstein GAN 的训练改进
损失函数: WGAN W 距离loss  考察真假样本的分布差异 判别器需要满足Lipschitz-1 Lipschitz-K 约束 所以不能加sigmoid约束范围
        通过对输入的求导 得到关于输入的导数 即权值W的函数 作为正则项  对其值进行直接约束 从而 满足Lipschitz-1 Lipschitz-K 约束
网络结构: MLP 至少有两层 即输入层后 至少1个中间层 然后是输出层 常用128节点 
数据形式: 不带卷积 没有深度维  图片压缩到0 1 之间 
生成器: sigmoid 映射到0 1 之间 迎合数据格式
判别器: 最后一层 没有sigmoid 没有relu 直接是matmul运算结果  迎合loss公式的约束 在全域内寻找满足Lipschitz-1 Lipschitz-K 约束的函数
初始化: xavier初始化  即考虑输入输出维度的 glorot uniform
训练： 判别器5次 生成器1次
"""
import my_mnist  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
(train_images,train_labels),(_, _) = my_mnist.load_data(get_new=False,
                                                        normalization=True,
                                                        one_hot=True,
                                                        detype=np.float32)
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
print(train_labels[0])
plt.imshow(train_images[0, :, :], cmap='gray')
plt.show()

class Dense(layers.Layer):
    def __init__(self, input_dim, units):
        super(Dense,self).__init__()
        # initializer = tf.initializers.glorot_uniform()
        initializer = tf.initializers.glorot_normal()
        self.w = tf.Variable(initial_value=initializer(shape=(input_dim,units),dtype=tf.float32),trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(1,units),dtype=tf.float32),trainable=True)#节点的偏置也是行向量 才可以正常计算 即对堆叠的batch 都是加载单个batch内
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.matmul(x,self.w)+self.b
            return y
        else:
            y = tf.matmul(x,self.w)+self.b
            return y 
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1 = Dense(28*28,128)
        self.dense2= Dense(128,32)
        self.dense3 = Dense(32,1)
    @tf.function
    def call(self,x,training=True):
        """
        batch*dim+batch*10 在index_1维度组合 其余维度不变
        """
        x = tf.reshape(x,[-1,784]) #reshape 不改变原始的元素顺序 这很重要 防止变形时变成转置 忽略batch大小  只关注后面的维度一致
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3_out = tf.nn.sigmoid(self.dense3(l2_out,training))
            return l3_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            # l3_out = tf.nn.sigmoid(self.dense3(l2_out,training))
            l3_out = self.dense3(l2_out,training)
            return l3_out
    @tf.function       
    def to_clip(self):
        for weight in self.trainable_variables:
            weight.assign(tf.clip_by_value(weight,clip_value_min=-0.01,clip_value_max=0.01))
d = Discriminator()
x = train_images[0:2, :, :]
print(d(x,training=False))#行向量统一输入  而batch是行向量在列方向堆叠后的矩阵 
print(len(d.trainable_variables))

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1 = Dense(100,32)
        self.dense2 = Dense(32,128)
        self.dense3 = Dense(128,784)
    @tf.function
    def call(self,x,training=True):
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.relu(self.dense2(l1_out,training))
            l3 = tf.nn.sigmoid(self.dense3(l2_out,training))
        l3_out = tf.reshape(l3,[-1,28,28])
        return l3_out

g = Generator()
z = tf.random.normal((1,100))
image = g(z,training=False)
plt.imshow(tf.reshape(image,(28,28)), cmap='gray')
plt.show()

print(d(image,training=False))

def d_loss(real_output, fake_output):
    total_loss = -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)#用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
    return total_loss
def g_loss(fake_output):
    return -tf.reduce_mean(fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0,beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,beta_1=0,beta_2=0.9)

EPOCHS = 400
BATCH_SIZE = 128
z_dim = 100
num_examples_to_generate = 100
seed = tf.random.normal([num_examples_to_generate, z_dim],mean=0.0,stddev=1.0)
# seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)

@tf.function
def D_train_step(images,labels):
    z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as disc_tape:
        generated_images = g(z,training=True)
        e = tf.random.uniform((images.shape[0],1,1),0.0,1.0)
        mid_images = e*images+(1-e)*generated_images
        with tf.GradientTape() as gradient_penalty:
            gradient_penalty.watch(mid_images)
            inner_loss = d(mid_images,training=True)
        penalty = gradient_penalty.gradient(inner_loss,mid_images)
        penalty_norm = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1)# 这是我自己认为的  因为只有梯度大于1的才需要优化哇
        # penalty_norm = 10.0*tf.math.square(tf.norm(penalty,ord='euclidean')-1) 这是按照算法愿意
        real_output = d(images,training=True)
        fake_output = d(generated_images,training=True)
        # disc_loss = d_loss(real_output,fake_output)
        disc_loss = d_loss(real_output,fake_output)+tf.reduce_mean(penalty_norm)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))

@tf.function
def G_train_step(images,labels):
    z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    # z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape:
        generated_images = g(z,training=True)
        fake_output = d(generated_images ,training=True)
        gen_loss = g_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
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
    predictions = model(test_input,training=False)
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*255.0, cmap='gray')
        plt.axis('off')
    plt.savefig('./WGAN_GP/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

print(time)
train(train_images,train_labels,EPOCHS)