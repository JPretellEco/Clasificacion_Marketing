# 📊 Estrategias Inteligentes de Marketing con Ciencia de Datos

## 🌍 Contexto Reconstruido: Del Caos al Conocimiento

En una era donde los clics narran historias más que las palabras, *ClickStream, una plataforma emergente en el mundo del e-commerce, se enfrenta a un desafío que muchas compañías digitales conocen bien: **¿cómo convertir un mar de datos en decisiones inteligentes?*

ClickStream acumuló miles de registros sobre sesiones, dispositivos, búsquedas y comportamientos de navegación de sus usuarios. Sin embargo, toda esa información dormía en silencio. El marketing seguía siendo genérico, sin alma, sin dirección. Fue entonces cuando se planteó una nueva estrategia: *entender para personalizar*. Nació así la necesidad de crear un sistema de segmentación de usuarios que permitiese campañas más certeras, empáticas y rentables.

Mi misión es convertir el ruido digital en una partitura de patrones que permita a ClickStream anticipar la intensidad de respuesta de cada usuario ante una campaña de marketing, clasificándolos en tres niveles: *bajo (1), medio (2) y alto (3)*.


## 🛠 Metodología Aplicada



### 1. 📚 Carga y Unificación de Datos

Se integraron datos del usuario como de su movimiento (user_train.csv, user_test.csv) con datos de comportamiento (session_train.csv, session_test.csv) mediante joins por user_id.

### 2. 🧪 Ingeniería de Características

A partir del análisis exploratorio, se construyeron nuevas variables:

* Número de sesiones y promedio de duración por usuario.
* Diversidad de dispositivos, países y navegadores utilizados.
* Generación demográfica inferida desde la edad (Gen Z, Millennial, Gen X, Boomer).

### 3. ⚖ Manejo del Desbalance

Se utilizó *SMOTE* para balancear las clases en la variable marketing_target, crucial para mejorar el recall y la generalización del modelo.

### 4. 🤖 Modelado

Se compararon tres algoritmos:

* *Random Forest* con max_depth=10, logrando el mejor balance entre bias y varianza.
* *Regresión Logística Multiclase*, que mostró robustez pero menor performance.
* *Árbol de Decisión*, con precisión aceptable pero más propenso al overfitting.

### 5. 📈 Validación

Se aplicó:

* *Train/test split (80/20)* con evaluación usando F1 Score ponderado.
* *Curva de aprendizaje* para verificar si el modelo se beneficiaba de más datos.
* *Validación cruzada (k=10)* para comprobar estabilidad.

### 6. 🧮 Predicción

Las predicciones sobre el dataset de prueba (X_test) fueron almacenadas como lo exige el desafío en:

json
{
  "target": {
    "297": 1,
    "11": 3,
    "67": 3,
    ...
  }
}


Guardadas dentro del directorio /predictions/predictions.json.

---

## 📊 Resultados Finales

| Modelo              | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Random Forest       | 0.87     | 0.88      | 0.87   | 0.87     |
| Logistic Regression | 0.73     | 0.73      | 0.73   |  0.73    |
| Decision Tree       | 0.80     | 0.80      | 0.80   | 0.80     |

El *Random Forest* demostró ser el mejor clasificador, destacando por su capacidad para capturar relaciones no lineales sin perder generalización. El modelo logró una *F1 ponderada de 0.87* en validación cruzada, superando el benchmark esperado.

---

## 🧭 Conclusión

De este modelo aprendí:
1. En producción, pueden darte ya la data entrenada, esto me ayudó a solo revisar los datos y generar nuevas variables para retroalimentar el nuevo modelo.
2. Hay que tener cuidado con el oversampling, cuando apliqué oversampling con boostrap, me generalizaba demasiado, al punto de obtener F1: 1.00
3. El ordern es muy importe, y la automatización también. En este modelo no utilicé pipelines, pero para otro modelo seré más organizado y haré pipelines con scikit-learn.

Lo técnico se fundió con lo humano: segmentar usuarios no es dividir, sino entender. 

---

