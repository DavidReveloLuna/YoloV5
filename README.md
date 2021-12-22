# YoloV5
Entrenaremos una red CNN Yolov5 para detectar armas

## 1. Etiquetado de dataset (labelImg)

    Usando conda en el entorno base, clonamos el repositorio de labelImg
    
    $ git clone https://github.com/tzutalin/labelImg
    $ conda install pyqt=5
    $ conda install -c anaconda lxml
    $ cd labelImg
    $ pyrcc5 -o libs/resources.py resources.qrc
    $ python labelImg.py
    
    Etiquetar el dataset y guardar usando la siguiente estructura:
    
    --train_data
      -- images
          --train
          --val
      -- labels
          --train
          --val

<img src="https://github.com/DavidReveloLuna/YoloV5/blob/master/assets/Labels.png" width="500">

## 2. Entrenamiento de la red 

Seguiremos la guía del sitio oficial [YoloV5](https://github.com/ultralytics/yolov5)
  
Sobre el codgio del enlace anterior haremos las siguientes modificaciones
      
    !unzip -q ../train_data.zip -d ../
    
Creamos un archivo customdata.yaml con la siguiente informacion, y guardar en la ruta yolov5/data
      
    path: ../train_data  # dataset root dir
    train: ../train_data/images/train/  # train images (relative to 'path') 128 images
    val: ../train_data/images/val/  # val images (relative to 'path') 128 images
    test:  # test images (optional)

    # Classes
    nc: 1  # number of classes
    names: ['gun']  # class names
    
Finalmente ejecutamos la linea de entrenamiento
  
    $  !python train.py --img 640 --batch 4 --epochs 100 --data customdata.yaml --weights yolov5s.pt --cache
    
## 3. Prueba en imagenes

    $ !python detect.py --weights /content/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source "../guntest.jpg"
    $ display.Image(filename='../guntest.jpg', width=600)

<img src="https://github.com/DavidReveloLuna/YoloV5/blob/master/assets/gundetection.jpg" width="500">
