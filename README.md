Выделение на фотографии человека и следующих атрибутов: принт кота, одежда в полоску, одежда в клетку, галстук, бабочка, очки.

### Установка и настройка

#### 1. Установить Anaconda
#### 2. Установить CUDA 9.0 с последними патчами, cuDNN версии 7
#### 3. Настроить виртуальное окружение

В Anaconda Prompt:
```
C:\> conda create -n tensorflow1 pip python=3.5
C:\> activate tensorflow1
(tensorflow1) C:\> python -m pip install --upgrade pip
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu==1.5.0
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python

(tensorflow1) C:\> set PYTHONPATH=PATH_TO_DIRECTORY_CONTAINING_object_detection;
```

### Запуск классификатора

В файле main_image.py изменить имя изображения IMAGE_NAME
```
(tensorflow1) C:PATH_TO_DIRECTORY_CONTAINING_object_detection\object_detection>python main_image.py
```

Для запуска на С++ слинковать с [Tensorflow C API](https://www.tensorflow.org/install/lang_c) и OpenCV.  
В файле main.cpp изменить путь до изображения и frozen_inference_graph.pb.
