import os  # Libreria para interactuar con el sistema operativo
import re  # Libreria para manejo de expresiones regulares
import json # Libreria para poder guardar en formato json las etiquetas con los nombres de las marcas
import numpy as np  # Libreria para cálculo numérico en Python
import PySimpleGUI as sg  # Libreria para crear interfaces gráficas de usuario
import tensorflow as tf  # Libreria para construir y entrenar modelos de deep learning
import urllib.request  # Libreria para trabajar con URLs y descargar archivos
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Funciones para cargar y procesar imágenes
from tensorflow.keras.applications.vgg16 import preprocess_input  # Función para preprocesamiento de imágenes
from tensorflow.keras.models import Sequential  # Clase para construir modelos secuenciales de Keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D  # Clases para construir capas de modelos de Keras
from tensorflow.keras.utils import to_categorical  # Función para transformar variables categóricas en variables numéricas
from keras.applications import VGG16  # Arquitectura de red neuronal pre-entrenada VGG16
from keras.optimizers import RMSprop  # Optimizador RMSprop para entrenamiento del modelo
from keras.layers import ZeroPadding2D  # Capa de cero relleno para imágenes
from tensorflow import keras
from PIL import Image  # Libreria para manipulación de imágenes en Python

global_marcas={}

#Esta extrae los datos de las imagenes de entrenamiento y validación desde las carpetas y las agrega a arreglos para después ser procesadas
def extraerDatosImagen(ruta_carpeta, cargarMarcas = False):
    lista_archivos = os.listdir(ruta_carpeta) # archivos de la carpeta de entrenamiento o validacion
    imagenes = [] #lista de imagenes
    etiquetas = [] #lista de etiquetas
    
    for carpeta in lista_archivos: #Se inicia un bucle 'for' para iterar por cada subcarpeta.
        if not carpeta.isnumeric():
            continue
        nombre_carpeta = int(carpeta)
        ruta_subcarpeta = os.listdir(ruta_carpeta+'/'+carpeta)
        for archivo in ruta_subcarpeta: #Se inicia un bucle 'for' para iterar por cada archivo en la subcarpeta.
            if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'): #Se verifica si el archivo termina con .jpg, .jpeg o .png.
                img = load_img(ruta_carpeta+'/'+carpeta+'/'+archivo, target_size=(224, 224)) #Se carga la imagen en la memoria con la biblioteca PIL y se cambia el tamaño a (224, 224).
                img = img_to_array(img) #Se convierte la imagen cargada en un array.
                img = preprocess_input(img)  #Se procesa la imagen para que tenga la misma forma que las imágenes utilizadas en el modelo VGG16.
                imagenes.append(img) #Se agrega la imagen procesada a la lista de imágenes.
                etiquetas.append(nombre_carpeta) #Se agrega la etiqueta de la imagen a la lista de etiquetas.
                if cargarMarcas:
                    global_marcas[str(nombre_carpeta)] = re.split('[\d.]', archivo)[0].replace('_',' ')

    imagenes = np.array(imagenes) #Se convierte la lista de imágenes a un array NumPy.
    etiquetas = np.array(etiquetas) #Se convierte la lista de etiquetas a un array NumPy.

    return [imagenes, etiquetas] #Se devuelve una lista que contiene el array de imágenes y el array de etiquetas.



def traerModeloConfigurado(cantidad_categorias): 

    # Define un modelo base pre-entrenado VGG16 para imágenes de entrada de 224x224 píxeles y 3 canales (RGB)
    modelo_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in modelo_base.layers: # bloquea todas las capas de este modelo base para que no se entrenen durante el entrenamiento de la red neuronal final
        layer.trainable = False
    
    modelo = Sequential() # crea un modelo de red neuronal secuencial vacío

    modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) # La entrada se define como una imagen de 224x224 píxeles y 3 canales (RGB)
    modelo.add(MaxPooling2D((2, 2), padding='same'))  # Agrega una capa de pooling con un tamaño de kernel de 2x2 y un padding 'same' para mantener la misma forma de salida que la entrada
    modelo.add(Conv2D(64, (3, 3), activation='relu')) # Agrega otra capa convolucional con 64 filtros y una función de activación ReLU
    modelo.add(MaxPooling2D((2, 2), padding='same')) # Agrega otra capa de pooling con un tamaño de kernel de 2x2 y un padding 'same'
    modelo.add(Conv2D(128, (3, 3), activation='relu')) # Agrega otra capa convolucional con 128 filtros y una función de activación ReLU
    modelo.add(MaxPooling2D((2, 2), padding='same'))  # Agrega otra capa de pooling con un tamaño de kernel de 2x2 y un padding 'same'
    modelo.add(Conv2D(128, (3, 3), activation='relu'))# Agrega otra capa convolucional con 128 filtros y una función de activación ReLU
    modelo.add(MaxPooling2D((2, 2), padding='same')) # Agrega otra capa de pooling con un tamaño de kernel de 2x2 y un padding 'same'
    modelo.add(Flatten()) # Aplana la salida de la última capa convolucional en una sola dimensión

    modelo.add(Dense(512, activation='relu')) # Agrega una capa densa con 512 neuronas y una función de activación ReLU

    
    modelo.add(Dense(cantidad_categorias, activation='softmax'))# Agrega una capa de salida con una cantidad de neuronas igual al número de categorías que se desean clasificar y una función de activación softmax para obtener probabilidades de pertenencia a cada categoría
    
    return modelo # dEvuelve el modelo de red neuronal completo


def procesarImagenes(imagenesEntrenamiento, imagenesValidacion):

    mean = np.mean(imagenesEntrenamiento) # Se calcula la media y la desviación estándar de las imágenes de entrenamiento
    std = np.std(imagenesEntrenamiento)

    # se nnormalizan las imágenes entre 0 y 1
    imagenesEntrenamiento = imagenesEntrenamiento / 255.0
    imagenesValidacion = imagenesValidacion / 255.0

    # Se retorna una lista con las imágenes de entrenamiento y validación normalizadas
    return [imagenesEntrenamiento, imagenesValidacion]


def procesarEtiquetas(etiquetasEntrenamiento, etiquetasValidacion):
    cantidad_categorias = len(set(etiquetasEntrenamiento)) # Se determina la cantidad total de categorías en las etiquetas de entrenamiento
    # Se utiliza to_categorical para convertir las etiquetas a vectores de ceros y unos
    etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento, cantidad_categorias)
    etiquetasValidacion = to_categorical(etiquetasValidacion, cantidad_categorias)

    # Se retornan las etiquetas procesadas, la cantidad de categorías
    return [etiquetasEntrenamiento, etiquetasValidacion, cantidad_categorias]

def compilarModelocompilarModelo(modelo):
    modelo.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

def compilarModelo(modelo):
    # Sse utiliza el optimizador RMSprop con una tasa de aprendizaje de 1e-4
    # Se utilizac la función de pérdida categorical_crossentropy ya que se trata de un problema de clasificación
    # se evaluará el modelo utilizando la métrica de precisión (accuracy)
    modelo.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


def definirModelo(): 
    # se obtienen las imágenes de entrenamiento, etiquetas de entrenamiento, imágenes de validación, etiquetas de validación, cantidad de categorías
    imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion, cantidad_categorias = traerDatosEntrenamiento()

    # se prepara el modelo con las imágenes y etiquetas obtenidas anteriormente
    return prepararModelo(imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion, cantidad_categorias)

def traerDatosEntrenamiento():
    # Cargar las imágenes y etiquetas de entrenamiento y validación
    imagenesEntrenamiento, etiquetasEntrenamiento = extraerDatosImagen('productos_pepsico_entrenamiento')
    imagenesValidacion, etiquetasValidacion = extraerDatosImagen('productos_pepsico_validacion', True)

    # Preprocesar las imágenes de entrenamiento y validación
    imagenesEntrenamiento, imagenesValidacion = procesarImagenes(imagenesEntrenamiento, imagenesValidacion)
    
    # Preprocesar las etiquetas de entrenamiento y validación
    etiquetasEntrenamiento, etiquetasValidacion, cantidad_categorias = procesarEtiquetas(etiquetasEntrenamiento, etiquetasValidacion)

    # Devolver los datos necesarios para entrenar el modelo
    return [imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion, cantidad_categorias]

def prepararModelo(imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion, cantidad_categorias):
    
    modelo = traerModeloConfigurado(cantidad_categorias)# Se define el modelo con la cantidad de categorías de los datos de entrenamiento
    compilarModelo(modelo) # Se compila el modelo con los parámetros adecuados
    entrenarModelo(modelo, imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion)  # Se entrena el modelo con los datos de entrenamiento y validación

    modelo.save('modelo.h5')

    # Se devuelve el modelo
    return modelo

def entrenarModelo(modelo, imagenesEntrenamiento, etiquetasEntrenamiento, imagenesValidacion, etiquetasValidacion): 
    history = modelo.fit(imagenesEntrenamiento, etiquetasEntrenamiento, epochs=10, batch_size=32, validation_data=(imagenesValidacion, etiquetasValidacion))

def procesarImagenSubida(rutaImagenSolicitada): 
   
    imagen = Image.open(rutaImagenSolicitada).resize((224, 224))  # Se carga la imagen desde la ruta especificada y se ajusta su tamaño
    imagen = np.array(imagen.convert('RGB')) # Se convierte la imagen a un arreglo de numpy en formato RGB
    salida = np.expand_dims(imagen, axis=0)  # Se agrega una dimensión extra al arreglo para representar un lote de un solo elemento
    salida = tf.keras.applications.mobilenet_v2.preprocess_input(salida) # Se normaliza la imagen utilizando la función de preprocesamiento de MobileNetV2

    return salida

# Esta funcion guarda las etiquetas después de haber compilado y guardado al modelo
def guardarEtiquetas(ruta_etiquetas, etiquetas):
    with open(ruta_etiquetas, 'w') as file:
        json.dump(etiquetas, file)

#Esta funcion verifica si es google colab para no mostrar el dialogo de subir imagen ya que este no lo soporta
def confirmarEntornoEjecucion(): 
    try:
        import google.colab #Pregunta si puede importar google colab
    except:
        return False

#Esta función mostrará un dialogo para facilitar la seleccion de la foto, no funcionara en google colab
def solicitarImagenPorDialogo(): 

    tiposImagen = ["Solo imagenes", "*.jpg *.png *.jpeg"] #Se establecen los tipos de imagen permitidos

    caja = [[sg.Image(key="-IMAGE-")], #Se especifican los componentes del dialogo
            [
                sg.Text("Seleccione la imagen del producto"), 
                sg.Input(size=(25, 1), key="-FILE-"),
                sg.FileBrowse(file_types=tiposImagen),
            ],
            [sg.Button("Subir Producto")]]

    ventana = sg.Window('Seleccione el producto', caja) #Se guarda la información del dialogo en una variable

    print(ventana) #Se muestra la ventana
    event, value = ventana.read() #Se captura los datos de la ruta de la imagen que se subió
    ventana.close()
 
    return value['-FILE-'] #retorna la ruta de la imagen

#Esta solicita la imagen por consola (para google colab)
def solicitarImagenManual(): 
    rutaImagenSolicitada = input('Ingrese la ruta de la imagen: ')
    return rutaImagenSolicitada

os.system('clear')


ruta_modelo = 'modelo.h5'
ruta_etiquetas = 'etiquetas.json'

modelo = None # Se inicializa la variable modelo que va a contener el modelo ya sea leyendolo o definiendolo y entrenandolo
etiquetas = None # Se inicializa la variable etiquetas que va a contener las marcas para poder dar una respuesta

if os.path.exists(ruta_modelo): # Se pregunta si ya existe un modelo guardado
    modelo = keras.models.load_model(ruta_modelo) # Se carga el modelo desde el directorio
else:
    modelo = definirModelo() # Se define el modelo
    guardarEtiquetas(ruta_etiquetas, global_marcas)


if not bool(global_marcas): # Se pregunta se cargaron las etiquetas durante el proceso de entrenamiento
    if os.path.exists(ruta_etiquetas): # Se pregunta si existe un archivo guardado con las etiquetas
        with open(ruta_etiquetas, 'r') as file: # Se lee el archivo
            etiquetas = json.load(file) # Se define el formato de las etiquetas
else:
    etiquetas = global_marcas # Se asignan las marcas a la variable etiquetas



# Se inicializa la variable de la ruta de la imagen que se subirá
rutaImagenSolicitada = ''

# Se verifica si el entorno de ejecución es de tipo manual o con diálogo
if (confirmarEntornoEjecucion()):
  rutaImagenSolicitada = solicitarImagenManual()
else:
  rutaImagenSolicitada = solicitarImagenPorDialogo()



# Se procesa la imagen y se realiza la predicción
salida = procesarImagenSubida(rutaImagenSolicitada)
prediccion = modelo.predict(salida)

# Se encuentra el índice de la marca con mayor probabilidad
index = np.argmax(prediccion)
os.system('clear')

# Se imprime el nombre de la marca correspondiente a la predicción

print('=============================================================')   
print('Este producto es:\n=> \033[1m',etiquetas[str(index)],'\033[0m')
print('=============================================================') 
print('\nSi no es el producto, puede ser alguno de los siguientes:') 

otrasPredicciones = np.argsort(prediccion)[0][::-1]

for  idx, indexPrediccion in enumerate(otrasPredicciones):
    if idx == 0:
        continue
    if idx > 5:
        break
    print('-> ',etiquetas[str(indexPrediccion)])