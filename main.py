from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import shutil
import matplotlib.pyplot as plt

app = Flask(__name__)

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path='brain_tumor_cnn.tflite')
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Diccionario de clases
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Carpeta de subida
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_with_tflite(img_array):
    # Preprocesar imagen para el modelo TFLite
    img_array = img_array.astype(np.float32) / 255.0
    
    # Configurar la entrada del modelo
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Ejecutar inferencia
    interpreter.invoke()
    
    # Obtener resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    probabilities = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Copiar imagen a carpeta static/uploads para previsualizar
            static_path = os.path.join('static', 'uploads', filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            shutil.copy(filepath, static_path)

            # Preprocesar imagen
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

            # Predicción con TFLite
            prediction = predict_with_tflite(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            prediction_result = f"Predicción: {predicted_class.upper()}"
            probabilities = {class_names[i]: float(f"{prob:.4f}") for i, prob in enumerate(prediction)}

            # Graficar probabilidades
            plt.figure(figsize=(6,4))
            plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
            plt.title('Probabilidades por clase')
            plt.ylabel('Confianza')
            plt.tight_layout()
            graph_path = os.path.join('static', 'uploads', 'probabilidades.png')
            plt.savefig(graph_path)
            plt.close()

    return render_template('index.html', prediction=prediction_result, image_name=filename, probs=probabilities)

if __name__ == '__main__':
    if not os.path.exists(app.config ['UPLOAD_FOLDER']):
        os.makedirs(app.config [ 'UPLOAD_FOLDER'])
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))
