#coding=utf8
import numpy as np
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adagrad, Adam
from keras.metrics import top_k_categorical_accuracy

DATA_NAME = 'homestyle'
NUM_CLASSES = 13
BATCH_SIZE = 128
outdir = sys.argv[1]
os.mkdir(outdir)

# 数据准备
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_dir = '/mnt/home/lixingjian/data/%s.split/train' % DATA_NAME
valid_dir = '/mnt/home/lixingjian/data/%s.split/valid' % DATA_NAME
train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                  target_size=(299,299),#Inception V3规定大小
                                  batch_size=BATCH_SIZE)
val_generator = val_datagen.flow_from_directory(directory=valid_dir,
                                target_size=(299,299),
                                batch_size=BATCH_SIZE)
train_steps = int(os.popen('find %s -name "*.jpg" |wc -l' % train_dir).read()) / BATCH_SIZE
valid_steps = int(os.popen('find %s -name "*.jpg" |wc -l' % valid_dir).read()) / BATCH_SIZE
print('train steps: %d, valid steps: %d' % (train_steps, valid_steps))

# 构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False)

# 增加新的输出层
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
predictions = Dense(NUM_CLASSES,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)

'''
这里的base_model和model里面的iv3都指向同一个地址
'''
'''
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy', top_k_categorical_accuracy])

setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    epochs=5,#2
                    validation_data=val_generator,
                    validation_steps=valid_steps,#12
                    class_weight='auto'
                    )
'''
def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 172 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.005),loss='categorical_crossentropy',metrics=['accuracy', top_k_categorical_accuracy])

setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=1,#train_steps,
                                 epochs=1,#0, 
                                 validation_data=val_generator,
                                 validation_steps=1,#valid_steps,
                                 class_weight='auto')

def out_mix_i(i):
    #mid_layer = Model(inputs=model.input, outputs=model.get_layer('mixed%d' % i).output)
    mid_layer = Model(inputs=model.input, outputs=model.layers[i].output)
    n = 0
    for xb, yb in train_generator:
        if n >= train_steps:
            break
        n += 1
        out = mid_layer.predict(xb)
        print(xb.shape)
        print(out.shape)
        np.save('%s/mixed.%d.%d.npy' % (outdir, i, n), out)
        np.save('%s/label.%d.%d.npy' % (outdir, i, n), yb)
out_mix_i(310)
out_mix_i(311)
out_mix_i(312)
out_mix_i(313)
