from flask import Flask, render_template,flash, request
from werkzeug.utils import secure_filename
import numpy as np
import shutil
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pandas as pd 

app = Flask(__name__)
# UPLOAD_FOLDER = r"C:\Users\adity\PycharmProjects\pythonProject\git\cnn_machine\ima"

UPLOAD_class_1 = '/home/adithya/Downloads/machine_cnn/class_1'
UPLOAD_class_2 = '/home/adithya/Downloads/machine_cnn/class_2'

app.secret_key = "Cairocoders-Ednalan"
app.config['UPLOAD_class_1'] = UPLOAD_class_1
app.config['UPLOAD_class_2'] = UPLOAD_class_2
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# def allowed_file(filename):
#  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['GET','POST'])
def upload_file():
 if request.method == 'POST':
        # check if the post request has the files part
  if 'files[]' not in request.files:
   flash('No file part')
   return redirect(request.url)
  files = request.files.getlist('files[]')
  for file in files:
   if file and file.filename:
    filename_batch1 = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_class_1'], filename_batch1))
  files2 = request.files.getlist('files2[]')
  for file in files2:
   if file and file.filename:
    filename_batch2 = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_class_2'], filename_batch2))
  flash('File(s) successfully uploaded')


  def num_files(x):
    for root, dirs, files in os.walk(x):
      for file in files:
          if file.endswith(".jpeg"):
              _, _, files = next(os.walk(root))
              no_of_files = len(files)
              no_of_files_test = no_of_files*0.8
              return no_of_files_test;

    #   for root, dirs, files in os.walk(UPLOAD_class_2):
    #       for file in files:
    #           if file.endswith(".jpg"):
    #               _, _, files = next(os.walk(root))
    #               no_of_files_2 = len(files)
    #               break;


  ## 1
  # Set the source and destination directories
  directory = "damaged"

  # Parent Directory path
  parent_dir_train = '/home/adithya/Downloads/machine_cnn/train'
  parent_dir_test = '/home/adithya/Downloads/machine_cnn/test'
  mode = 0o777
  path_dr = os.path.join(parent_dir_train, directory)
  os.mkdir(path_dr,mode)



  src_dir = UPLOAD_class_1
  dst_dir = path_dr

  # Set the number of images to move
  #   num_images = int(no_of_files_1*0.8)

  # Get a list of all image files in the source directory
  image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.jpg')]

  # Move the first `num_images` images to the destination directory
  for i, image_file in enumerate(image_files):
      if i >= num_files(src_dir):
          break
      shutil.move(os.path.join(src_dir, image_file), dst_dir)

  ## 1.5

  mode = 0o777
  path_de = os.path.join(parent_dir_test, directory)
  os.mkdir(path_de,mode)


  src_dir = UPLOAD_class_1
  dst_dir = path_de

  # Set the number of images to move

  # Get a list of all image files in the source directory
  image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpeg') or f.endswith('.png')or f.endswith('.jpg')]

  # Move the first `num_images` images to the destination directory
  for i, image_file in enumerate(image_files):
      shutil.move(os.path.join(src_dir, image_file), dst_dir)




  ## 2
  # Set the source and destination directories
  directory = "perfect"

  # Parent Directory path
  mode = 0o777
  path_dr = os.path.join(parent_dir_train, directory)
  os.mkdir(path_dr,mode)



  src_dir = UPLOAD_class_2
  dst_dir = path_dr

  # Set the number of images to move
  #   num_images = int(no_of_files_2*0.8)

  # Get a list of all image files in the source directory
  image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpeg') or f.endswith('.png')or f.endswith('.jpg')]

  # Move the first `num_images` images to the destination directory
  for i, image_file in enumerate(image_files):
      if i >= num_files(src_dir):
          break
      shutil.move(os.path.join(src_dir, image_file), dst_dir)

  ## 2.5

  mode = 0o777
  path_de = os.path.join(parent_dir_test, directory)
  os.mkdir(path_de,mode)


  src_dir = UPLOAD_class_2
  dst_dir = path_de

  # Set the number of images to move

  # Get a list of all image files in the source directory
  image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpeg') or f.endswith('.png')or f.endswith('.jpg')]

  # Move the first `num_images` images to the destination directory
  for i, image_file in enumerate(image_files):
      shutil.move(os.path.join(src_dir, image_file), dst_dir)


  train_path = '/home/adithya/Downloads/machine_cnn/train'
  test_path = '/home/adithya/Downloads/machine_cnn/test'

  image_gen = ImageDataGenerator(rescale=1/255, 
                                zoom_range=0.1, 
                                brightness_range=[0.9,1.0])

  image_shape = (300,300,1) 
  batch_size = 32

  train_set = image_gen.flow_from_directory(train_path,
                                              target_size=image_shape[:2],
                                              color_mode="grayscale",
                                              classes={'damaged': 0, 'perfect': 1},
                                              batch_size=batch_size,
                                              class_mode='binary',
                                              shuffle=True,
                                              seed=0)

  test_set = image_gen.flow_from_directory(test_path,
                                            target_size=image_shape[:2],
                                            color_mode="grayscale",
                                            classes={'damaged': 0, 'perfect': 1},
                                            batch_size=batch_size,
                                            class_mode='binary',
                                            shuffle=False,
                                            seed=0)

  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=(7,7), strides=2, input_shape=image_shape, activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, input_shape=image_shape, activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=image_shape, activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Flatten())
  model.add(Dense(units=224, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(units=1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  model_save_path = 'lki.hdf5'
  early_stop = EarlyStopping(monitor='val_loss',patience=2)
  checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=1, save_best_only=True, monitor='val_loss')

  n_epochs = 4
  results = model.fit_generator(train_set, epochs=n_epochs, validation_data=test_set, callbacks=[early_stop,checkpoint],verbose=0)
  results_1 = max(results.history['val_accuracy'])
  final_results = round(results_1,2)
  results_2 = max(results.history['val_loss'])
  final_results_2 = round(results_2,2)
  return render_template('predict.html', status = final_results, status_2 = final_results_2)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5000)