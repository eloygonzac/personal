# main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# 1. CREACIÓN DE LA INSTANCIA DE LA APLICACIÓN
# La variable 'app' es la instancia principal de FastAPI.
app = FastAPI(
    title="Servicio de Predicción de Precios ML",
    description="API que utiliza un Pipeline de scikit-learn (preprocesamiento + Regresión Lineal) para predecir precios.",
    version="1.0.0"
)

# Variable global para almacenar el pipeline del modelo cargado
pipeline_modelo = None 
RUTA_MODELO = "pipeline_modelo_produccion.joblib"

# --- DEFINICIÓN DE CLASES ---

# 2. CONTRATO DE DATOS (PYDANTIC)
# Esta clase define y valida la estructura exacta de los datos de entrada.
class DatosEntrada(BaseModel):
    # La edad debe ser un número entero (int)
    edad: int = Field(..., description="Edad del cliente o del objeto, debe ser un número entero positivo.")
    
    # La ubicación debe ser una cadena de texto (str)
    ubicacion: str = Field(..., description="Categoría de ubicación (ej. 'A', 'B', 'C').")

    # Configuración de Pydantic para mostrar un ejemplo en la documentación /docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "edad": 35,
                    "ubicacion": "B"
                }
            ]
        }
    }

# --- HOOK DE INICIALIZACIÓN ---

# 3. CARGA DEL MODELO AL INICIO DEL SERVIDOR
# Esta función se ejecuta *una sola vez* cuando el servidor se enciende (startup).
@app.on_event("startup")
def cargar_modelo():
    global pipeline_modelo
    
    # Verificamos si el archivo existe
    if not os.path.exists(RUTA_MODELO):
        print(f"ERROR: No se encontró el archivo del modelo en la ruta: {RUTA_MODELO}")
        # En un entorno real, la app debe fallar si el modelo no existe
        return

    try:
        pipeline_modelo = joblib.load(RUTA_MODELO)
        print(f"✅ Modelo cargado exitosamente desde: {RUTA_MODELO}")
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo: {e}")
        # Si el modelo no carga, detenemos la app
        raise RuntimeError("El servidor no puede arrancar sin un modelo válido.")

# --- ENDPOINT DE PREDICCIÓN ---

# 4. DEFINICIÓN DEL ENDPOINT POST /predict
@app.post("/predict", summary="Realiza una predicción de precio", response_description="El precio estimado por el modelo.")
def predict(data: DatosEntrada):
    """
    Recibe la edad y la ubicación, aplica el preprocesamiento definido en el
    Pipeline y devuelve el precio predicho.
    """
    
    # 5. CONVERSIÓN DE DATOS Y PREDICCIÓN
    
    # Convertir los datos de Pydantic validados a un DataFrame de Pandas.
    # Es VITAL que los nombres de las columnas ('Edad', 'Ubicacion') coincidan 
    # con los nombres que usaste para entrenar el modelo.
    datos_df = pd.DataFrame({
        'Edad': [data.edad],
        'Ubicacion': [data.ubicacion]
    })
    
    # Realizar la predicción usando el pipeline completo
    # El pipeline aplica StandardScaler y OneHotEncoder automáticamente.
    prediccion = pipeline_modelo.predict(datos_df)
    
    # El resultado es un array numpy; lo convertimos a un flotante estándar 
    precio_predicho = float(prediccion[0])
    
    # Devolver la respuesta en formato JSON
    return {"precio_predicho": round(precio_predicho, 2)}
