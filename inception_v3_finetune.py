#coding=utf8
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adagrad, Adam
from keras.metrics import top_k_categorical_accuracy

DATA_NAME = sys.argv[1]
feature_dir = sys.argv[2]
data_dir = '/home/work/lixingjian/data/custom_data'
train_dir = '%s/%s.split/train' % (data_dir, DATA_NAME)
valid_dir = '%s/%s.split/valid' % (data_dir, DATA_NAME)
NUM_CLASSES = int(os.popen('ls train_dir |wc -l'))
BATCH_SIZE = 128

# 数据准备
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

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

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 172 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.005),loss='categorical_crossentropy',metrics=['accuracy', top_k_categorical_accuracy])

setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_steps,
                                 epochs=5, 
                                 validation_data=val_generator,
                                 validation_steps=valid_steps,
                                 class_weight='auto')

def write_features(i, generator, steps, feature_dir):
    os.system('mkdir -p ' + feature_dir)
    mid_layer = Model(inputs=model.input, outputs=model.layers[i].output)
    n = 0
    for xb, yb in generator:
        if n >= steps:
            break
        n += 1
        out = mid_layer.predict(xb)
        np.save('%s/mixed.%d.%d.npy' % (feature_dir, i, n), out)
        np.save('%s/label.%d.%d.npy' % (feature_dir, i, n), yb)

write_features(311, train_generator, train_steps, feature_dir + '/train')  #feature layer output
write_features(311, val_generator, valid_steps, feature_dir + '/test')  #feature layer output

