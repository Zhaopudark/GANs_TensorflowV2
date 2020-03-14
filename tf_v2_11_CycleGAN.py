from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import my_horse2zebra
import numpy as np
# tfds.disable_progress_bar()
# AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset, metadata = tfds.load('cycle_gan/horse2zebra',
#                               with_info=True, as_supervised=True)

# train_horses, train_zebras = dataset['trainA'], dataset['trainB']
# test_horses, test_zebras = dataset['testA'], dataset['testB']

(train_horses,train_zebras),(test_horses,test_zebras) = my_horse2zebra.load_horse2zebra("./datasets/horse2zebra/",get_new=False,detype=np.int16)
# 先转成有符号的 不能转成int8 因为int8 必然是会截断首位 做符号位的
print(train_horses.shape)
plt.imshow(train_horses[0, :, :,:])
plt.show()
a = train_horses[0, :, :]
plt.hist(a.flatten(), bins=80, color='c')
plt.xlabel("Pix 0 ~ 255")
plt.ylabel("Frequency")
plt.show()
a = a/127.5-1.0
print(a.dtype)
plt.imshow(a) # 它只能识别0~255的整型 0~1的浮点型 两种 如果是浮点型 自动截取0~1.0区间 如果是整形 截取0~255区间 所以归一化后显示异常时正常现象
plt.show()
plt.hist(a.flatten(), bins=80, color='c')
plt.xlabel("Pix -1 ~ 1")
plt.ylabel("Frequency")
plt.show()

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image
# 将图像归一化到区间 [-1, 1] 内。
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
def random_jitter(image):
    # 调整大小为 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method="nearest")
                            # 最近邻插值 获得图片的缩放比gama_x=缩放前x/缩放后x gama_y=缩放前y/缩放后y 
                            # 对于插值后图像的 x位置像素 = x*gama_x位置的原图像像素 四舍五入取最近的位置即可 
                            # 图像插值(一般是放大)之我见  有了缩放比之后 1*缩放比 形成小于1的间隔 以这个间隔 形成浮点坐标 即带入计算的差指点 计算完所有浮点坐标后  即计算完了整个插值后的整数坐标位置的像素值 
                            # 浮点坐标是在原始图像坐标系中的 而浮点坐标一一对应插值后的图像的像素点
                            # 所以 最近邻插值 浮点坐标的像素是取与其最近的原始图像坐标的像素值 
                            #      双线性插值 浮点坐标的像素是取与其最近的四个原始图像坐标线性计算而来 即先横向线性插值 再纵向线性插值
                            # 这种浮点坐标的概念有利于理解插值过程
                            # 如果是缩小 则缩放比大于1 则也类似的
    # 随机裁剪到 256 x 256 x 3
    image = random_crop(image)
    # 随机镜像
    image = tf.image.random_flip_left_right(image)

    return image
def preprocess_image_train(image, label=None):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    image = normalize(image)
    return image

train_horses = tf.map_fn(preprocess_image_train,train_horses.astype(np.float32))
print(train_horses)
plt.imshow(train_horses[0, :, :,:])
plt.show()
a = train_horses.numpy() #numpy 是tensor的一个方法  调用numpy() 方法 返回一个numpy矩阵

plt.hist(a[0,:,:,:].flatten(), bins=80, color='c')
plt.xlabel("Pix 0 ~ 255")
plt.ylabel("Frequency")
plt.show()
train_zebras = tf.map_fn(preprocess_image_train,train_zebras.astype(np.float32))
test_horses = tf.map_fn(preprocess_image_train,test_horses.astype(np.float32))
test_zebras = tf.map_fn(preprocess_image_train,test_zebras.astype(np.float32))

train_horses=tf.data.Dataset.from_tensor_slices(train_horses).shuffle(BUFFER_SIZE).batch(1)
train_zebras=tf.data.Dataset.from_tensor_slices(train_zebras).shuffle(BUFFER_SIZE).batch(1)
test_horses=tf.data.Dataset.from_tensor_slices(test_horses).shuffle(BUFFER_SIZE).batch(1)
test_zebras=tf.data.Dataset.from_tensor_slices(test_zebras).shuffle(BUFFER_SIZE).batch(1)

"""
tf.data.Dataset.from_tensor_slices 构建的是一个自己的实例 既可以list转为列表 每个元素都是一个batch 
也可以用 iter() 构建迭代器
"""
print("**************************************")
sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))
plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
plt.show()

plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
plt.show()

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_zebra = generator_g(sample_horse)
to_horse = generator_f(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

plt.show()

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5
def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 40
def generate_images(model, test_input):
    prediction = model(test_input)
        
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # 获取范围在 [0, 1] 之间的像素值以绘制它。
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def train_step(real_x, real_y):
    # persistent 设置为 Ture，因为 GradientTape 被多次应用于计算梯度。
    with tf.GradientTape(persistent=True) as tape:
        # 生成器 G 转换 X -> Y。
        # 生成器 F 转换 Y -> X。
        
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x 和 same_y 用于一致性损失。
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # 计算损失。
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        
        # 总生成器损失 = 对抗性损失 + 循环损失。
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
    # 计算生成器和判别器损失。
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)
    
    # 将梯度应用于优化器。
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                generator_f.trainable_variables))
    
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))
    
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))

                    
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n+=1

    clear_output(wait=True)
    # 使用一致的图像（sample_horse），以便模型的进度清晰可见。
    generate_images(generator_g, sample_horse)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


# 在测试数据集上运行训练的模型。
for inp in test_horses.take(5):
    generate_images(generator_g, inp)