# Usa una imagen base oficial de Python
FROM python:latest

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requerimientos en el directorio de trabajo
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación en el directorio de trabajo
COPY . .

# Expone el puerto 5050
EXPOSE 5050

# Define el comando de inicio
CMD ["python", "api.py"]
