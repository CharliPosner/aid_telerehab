import os
import glob 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import deepdish as dd

from functools import partial

import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers.core import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight


###########################################################################


########## SET RANDOM SEEDS ##########

rdm = 1
random.seed(rdm)
np.random.seed(rdm)
tf.random.set_seed(rdm)


########## CONFIGURE PARAMETERS ##########

test_split = 0.3

img_height = 224
img_width = 224

epochs = 500

batch_size = 8

learning_rate = 0.00001
lr_sci_notation = "{:.0E}".format(learning_rate)

rating_layer = 15
exercise_layer = 15

shuffle = True
trainable = True


###########################################################################


########## LOAD MOVI MODEL ##########

model = keras.models.load_model('movi224normBP__epoch0011__loss0.3030__lr5E-05.hdf5', compile=False)
#model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy']) ## --> if above edit doesn't work, uncomment this line
model.summary()

pretrained_model = keras.Model(model.inputs, model.layers[-3].output)
layer_names = ['image_input', 'resnet50', 'flatten']
for i, layer in enumerate(pretrained_model.layers):
    layer._name = layer_names[i]
    layer.trainable = trainable 
pretrained_model.summary()


########## CREATE KIMORE MODEL ##########

flatten = pretrained_model.output

dense_rating = Dense(rating_layer, activation='relu', name=f'dense1_{rating_layer}_rating')(flatten)
rating_output = Dense(3, activation='softmax', name='rating_output')(dense_rating)

dense_exercise = Dense(exercise_layer, activation='relu', name=f'dense1_{exercise_layer}_exercise')(flatten)
exercise_output = Dense(5, activation='softmax', name='exercise_output')(dense_exercise)

kimore_model = keras.Model(inputs=pretrained_model.input, outputs=[rating_output, exercise_output])
kimore_model._name = "kimore_model"
kimore_model.summary()


###########################################################################


########## LOAD DATA ##########

with open("/user/work/le19806/kimore224norm_blazepose_xyz_filenames.txt", "r") as txt_file:
    file_contents = txt_file.readlines()
kimore_filenames = file_contents[0].split(' ')[:-1]

kimore_dicts = dd.io.load('/user/work/le19806/kimore224norm_blazepose_xyz_dicts.h5')


########## NEW TEST SPLIT ##########

subject_list = []
for dic in kimore_dicts:
    group = dic['group']
    sub_id = dic['subject_id']
    subject = [group, sub_id]
    subject_list.append(subject)
    
unique_subjects = [list(x) for x in set(tuple(x) for x in subject_list)]

split = 1
while (split > test_split):
    test_subject_num = round(0.1 * len(unique_subjects))
    test_subject_list = np.array(random.sample(unique_subjects, test_subject_num))
    test_dicts = []
    test_idx = []
    for d in range(len(kimore_dicts)):
        group = kimore_dicts[d]['group']
        sub_id = kimore_dicts[d]['subject_id']
        if (group in test_subject_list[:, 0]):
            if (sub_id in test_subject_list[:, 1]):
                test_dicts.append(kimore_dicts[d])
                test_idx.append(d)          
                
    split = len(test_dicts) / len(kimore_dicts)

print(len(test_dicts))
print(len(test_dicts) / len(kimore_dicts))

non_test_idx = [i for i in list(range(5809)) if i not in test_idx]

test_files = [kimore_filenames[i] for i in test_idx]
non_test_files = [kimore_filenames[i] for i in non_test_idx]

train_file_num = round(0.8 * len(non_test_files))
train_files = random.sample(non_test_files, train_file_num)
train_idx = [i for i in range(len(non_test_files)) if non_test_files[i] in train_files]

val_files = [f for f in non_test_files if f not in train_files]
val_idx = [i for i in range(len(non_test_files)) if non_test_files[i] in val_files]

total = len(test_files) + len(train_files) + len(val_files)
print(total)


########## GET TRAIN DATA ##########

train_input_list = []
train_ratings_list = []
train_exercises_list = []

for idx in train_idx:
    train_dict = kimore_dicts[idx]
    train_input_list.append(train_dict["image"])
    train_ratings_list.append(train_dict["rating_onehot"])
    train_exercises_list.append(train_dict["exercise_onehot"])

train_inputs = np.stack(train_input_list)
train_ratings = np.stack(train_ratings_list)
train_exercises = np.stack(train_exercises_list)


########## GET VALIDATION DATA ##########

val_input_list = []
val_ratings_list = []
val_exercises_list = []

for idx in val_idx:
    val_dict = kimore_dicts[idx]
    val_input_list.append(val_dict["image"])
    val_ratings_list.append(val_dict["rating_onehot"])
    val_exercises_list.append(val_dict["exercise_onehot"])

val_inputs = np.stack(val_input_list)
val_ratings = np.stack(val_ratings_list)
val_exercises = np.stack(val_exercises_list)


########## GET TEST DATA ##########

test_input_list = []
test_ratings_list = []
test_exercises_list = []

for idx in test_idx:
    test_dict = kimore_dicts[idx]
    test_input_list.append(test_dict["image"])
    test_ratings_list.append(test_dict["rating_onehot"])
    test_exercises_list.append(test_dict["exercise_onehot"])

test_inputs = np.stack(test_input_list)
test_ratings = np.stack(test_ratings_list)
test_exercises = np.stack(test_exercises_list)



###########################################################################


########## GET RATING INFORMATION ##########

rating_onehot_list = []
rating_list = []
for dictionary in kimore_dicts:
  rating_onehot_list.append(dictionary["rating_onehot"])
  rating_list.append(dictionary["rating"])

rating_onehot_ndarray = np.stack(rating_onehot_list)
rating_class_list = sorted([i-1 for i in rating_list])


########## COMPUTE CLASS WEIGHTS ##########

sklearn_class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(rating_class_list), y= rating_class_list)

w_array = np.ones((3, 3))
w_array[0, :] = sklearn_class_weights[0]
w_array[1, :] = sklearn_class_weights[1]
w_array[2, :] = sklearn_class_weights[2]


########## CREATE NEW LOSS CLASS FOR WEIGHTED RATING CLASSIFICATION ##########

class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    
  def __init__(self, cost_mat, name='weighted_categorical_crossentropy', **kwargs):
    cost_mat = np.array(cost_mat)   
    assert(cost_mat.ndim == 2)
    assert(cost_mat.shape[0] == cost_mat.shape[1])
    super().__init__(name=name, **kwargs)
    self.cost_mat = K.cast_to_floatx(cost_mat)


  def __call__(self, y_true, y_pred, sample_weight=None):
    assert sample_weight is None, "should only be derived from the cost matrix"  
    return super().__call__(
        y_true = y_true, 
        y_pred = y_pred, 
        sample_weight = get_sample_weights(y_true, y_pred, self.cost_mat),
    )


  def get_config(self):
    config = super().get_config().copy()
    config.update({'cost_mat': (self.cost_mat)})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)

    y_pred.shape.assert_has_rank(2)
    assert(y_pred.shape[1] == num_classes)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred = K.one_hot(K.argmax(y_pred), num_classes)

    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n


tf.keras.losses.WeightedCategoricalCrossentropy = WeightedCategoricalCrossentropy


###########################################################################


########## CONFIGURE FILE NAME INFORMATION ##########

file_location = '/user/home/le19806/'
train_filename = 'kimoreTF_movi5E-05_15N_test3E-1'

checkpoint_filename = 'kimoreTF_movi5E-05_15N_test3E-1'

train_info = train_filename + '.png'


########## COMPILE MODEL ##########

kimore_model.compile(
    optimizer = Adam(learning_rate=learning_rate),
    loss = {
        'rating_output': WeightedCategoricalCrossentropy(w_array), 
        'exercise_output': 'categorical_crossentropy'},
    metrics = {
        'rating_output': ['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')], 
        'exercise_output':['accuracy']})


########## CREATE MODEL CHECKPOINT CALLBACK ##########

checkpoint_filepath = file_location + checkpoint_filename

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = False,
    monitor = 'val_rating_output_loss',
    mode = 'min',
    save_best_only = True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_exercise_output_loss",
    min_delta=0,
    patience=20,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0)

########## FIT MODEL ##########

history = kimore_model.fit(
    x = train_inputs,
    y = [train_ratings, train_exercises],
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [model_checkpoint_callback, early_stopping_callback],
    validation_data = (val_inputs, [val_ratings, val_exercises]),
    shuffle = shuffle)

history_filepath = file_location + 'history_' + train_filename + '.npy'
np.save(history_filepath, history.history)

###########################################################################


########## RATING ACCURACY GRAPH ##########

fig1 = plt.gcf
plt.plot(history.history['rating_output_accuracy'])
plt.plot(history.history['val_rating_output_accuracy'])
plt.grid()
plt.title('Rating Accuracy TF')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
rating_accuracy_fname = file_location + 'ratingAccuracy_' + train_info
plt.savefig(rating_accuracy_fname)
plt.close()


########## RATING PRECISION GRAPH ##########

fig2 = plt.gcf
plt.plot(history.history['rating_output_precision'])
plt.plot(history.history['val_rating_output_precision'])
plt.grid()
plt.title('Rating Precision TF')
plt.ylabel('Precision')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
rating_precision_fname = file_location + 'ratingPrecision_' + train_info
plt.savefig(rating_precision_fname)
plt.close()


########## RATING RECALL GRAPH ##########

fig3 = plt.gcf
plt.plot(history.history['rating_output_recall'])
plt.plot(history.history['val_rating_output_recall'])
plt.grid()
plt.title('Rating Recall TF')
plt.ylabel('Recall')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
rating_recall_fname = file_location + 'ratingRecall_' + train_info
plt.savefig(rating_recall_fname)
plt.close()


########## RATING LOSS GRAPH ##########

fig4 = plt.gcf()
plt.plot(history.history['rating_output_loss'])
plt.plot(history.history['val_rating_output_loss'])
plt.grid()
plt.title('Rating Loss TF')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
rating_loss_fname = file_location + 'ratingLoss_' + train_info
plt.savefig(rating_loss_fname)
plt.close()


########## EXERCISE ACCURACY GRAPH ##########

fig5 = plt.gcf()
plt.plot(history.history['exercise_output_accuracy'])
plt.plot(history.history['val_exercise_output_accuracy'])
plt.grid()
plt.title('Exercise Accuracy TF')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
exercise_accuracy_fname = file_location + 'exerciseAccuracy_' + train_info
plt.savefig(exercise_accuracy_fname)
plt.close()

########## EXERCISE LOSS GRAPH ##########

fig6 = plt.gcf()
plt.plot(history.history['exercise_output_loss'])
plt.plot(history.history['val_exercise_output_loss'])
plt.grid()
plt.title('Exercise Loss TF')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
exercise_loss_fname = file_location + 'exerciseLoss_' + train_info
plt.savefig(exercise_loss_fname)
plt.close()


###########################################################################


########## LOAD TRAINED MODEL FROM CHECKPOINT ##########

model = tf.keras.models.load_model(
    checkpoint_filepath,
    compile = True,
    custom_objects = {'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy(w_array)})


########## TEST RATING RESULTS ##########

test_y_pred = kimore_model.predict(test_inputs)
predicted_ratings = test_y_pred[0]

test_rating_report = classification_report(test_ratings.argmax(axis=1), predicted_ratings.argmax(axis=1), output_dict=True)
tr_df = pd.DataFrame(test_rating_report).transpose()
test_rating_report_fname = file_location + 'testRatingReport__' + train_filename + '.csv'
tr_df.to_csv(test_rating_report_fname)

test_matrix = ConfusionMatrixDisplay(confusion_matrix(test_ratings.argmax(axis=1), predicted_ratings.argmax(axis=1), normalize='true'))
test_matrix.plot()
test_rating_matrix_fname = file_location + 'testRatingMatrix__' + train_filename + '.png'
plt.savefig(test_rating_matrix_fname)
plt.close()


########## TEST EXERCISE RESULTS ##########

test_y_pred = kimore_model.predict(test_inputs)
predicted_exercises = test_y_pred[1]

test_exercise_report = classification_report(test_exercises.argmax(axis=1), predicted_exercises.argmax(axis=1), output_dict=True)
te_df = pd.DataFrame(test_exercise_report).transpose()
test_exercise_report_fname = file_location + 'testExerciseReport__' + train_filename + '.csv'
te_df.to_csv(test_exercise_report_fname)

test_matrix = ConfusionMatrixDisplay(confusion_matrix(test_exercises.argmax(axis=1), predicted_exercises.argmax(axis=1), normalize='true'))
test_matrix.plot()
test_exercise_matrix_fname = file_location + 'testExerciseMatrix__' + train_filename + '.png'
plt.savefig(test_exercise_matrix_fname)
plt.close()