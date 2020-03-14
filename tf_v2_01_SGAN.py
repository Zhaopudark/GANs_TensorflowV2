from __future__ import absolute_import, division, print_function, unicode_literals
"""
基于tensorflow 高阶API
SGAN  标准GAN  standard GAN 
损失函数: 经典log损失 即交叉熵损失 对抗损失 赋予正负样本 1 0 标签后的推导值 其实就是交叉熵 只是分开了正负样本
        但是基于tensorflow的交叉熵计算时优化过的 对于log(0)不会出现无穷值 这就是我自己的log函数容易崩坏的原因
网络结构: MLP 至少有两层 即输入层后 至少1个中间层 然后是输出层 常用128节点 
数据形式: 不带卷积 没有深度维  图片压缩到0 1 之间 
生成器: sigmoid 映射到0 1 之间 迎合数据格式
判别器: sigmoid 映射到0 1 之间 迎合loss公式的约束
初始化: xavier初始化  即考虑输入输出维度的 glorot uniform
训练： 判别器和生成器同时训练 同步训练 不偏重任一方
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
        initializer = tf.initializers.glorot_uniform()
        # initializer = tf.initializers.glorot_normal()
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

x = tf.random.normal(shape=(64,784))
a = Dense(28*28,128)
print(a(x))
print(len(a.trainable_variables))

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1 = Dense(28*28,128)
        self.dense2 = Dense(128,1)
    @tf.function
    def call(self,x,training=True):
        x = tf.reshape(x,[-1,784]) #reshape 不改变原始的元素顺序 这很重要 防止变形时变成转置 忽略batch大小  只关注后面的维度一致
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out

d = Discriminator()
x = train_images[0:2, :, :]
print(d(x,training=False))#行向量统一输入  而batch是行向量在列方向堆叠后的矩阵 
print(len(d.trainable_variables))

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1 = Dense(100,128)
        self.dense2 = Dense(128,784)
    @tf.function
    def call(self,x,training=True):
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        l2_out = tf.reshape(l2,[-1,28,28])
        return l2_out

g = Generator()
z = tf.random.normal((1,100))

image = g(z,training=False)
plt.imshow(tf.reshape(image,(28,28)), cmap='gray')
plt.show()

print(d(image,training=False))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def d_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def g_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 200
BATCH_SIZE = 128
z_dim = 100
num_examples_to_generate = 100
# seed = tf.random.normal([num_examples_to_generate, z_dim],mean=0.0,stddev=1.0)
seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)



@tf.function
def train_step(images,labels):
    # z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(z,training=True)
        real_output = d(images,training=True)
        fake_output = d(generated_images,training=True)
        gen_loss = g_loss(fake_output)
        disc_loss = d_loss(real_output,fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))

def train(train_images,train_labels,epochs):
    index = list(range(train_images.shape[0]))
    np.random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
    for epoch in range(epochs):
        start = time.time()
        while True:
            try:
                x_real_bacth = next(images_batches)
                y_label_bacth = next(labels_batches)
                train_step(x_real_bacth,y_label_bacth)
            except StopIteration:
                del images_batches
                del labels_batches
                np.random.shuffle(index)
                train_images = train_images[index]
                train_labels = train_labels[index]
                images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
                labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
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
    plt.savefig('./SGAN/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

print(time)
train(train_images,train_labels,EPOCHS)