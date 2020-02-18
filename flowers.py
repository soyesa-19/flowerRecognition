from keras.applications import VGG16
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img_rows = 224
img_cols = 224

vgg16 = VGG16(
    weights = "imagenet",
    include_top=False,
    input_shape=(img_rows, img_cols,3)
)

#  Freeze the low level layers
for layer in vgg16.layers:
    layer.trainable = False

# for (i,layer) in enumerate(vgg16.layers):
#     print(str(i)+" "+ layer.__class__.__name__, layer.trainable)

def topLayers(bottomlayer, num_classes, D=256):

    top_model = bottomlayer.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation="softmax")(top_model)
    return top_model

num_classes=17
FC_head = topLayers(vgg16, num_classes)

model = Model(inputs = vgg16.input, outputs= FC_head)

# print(model.summary())

train_dir = "./17_flowers/train"
test_dir = "./17_flowers/validation"

train_datagen = ImageDataGenerator(
    rescale= 1. /255,
    rotation_range= 20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1. /255)

train_batchsize = 16
test_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size= train_batchsize,
    class_mode = "categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size= test_batchsize,
    class_mode="categorical",
    shuffle= False
)

modelcheckpoint = ModelCheckpoint(
    "flower.h5",
    monitor="val_loss",
    mode="min",
    verbose =1,
    save_best_only=True
    )

earlystopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights= True
)

reducelr = ReduceLROnPlateau(
    monitor="val_loss",
    factor= 0.2,
    patience=3,
    verbose=1,
    min_delta=0.00001

)

callbacks = [modelcheckpoint, earlystopping, reducelr]

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"])

nb_train = 1190
nb_test = 170
batch_size =16
epochs = 25

model.fit(
    train_generator,
    steps_per_epoch= nb_train // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_steps= nb_test // batch_size,
    validation_data= test_generator    
)
