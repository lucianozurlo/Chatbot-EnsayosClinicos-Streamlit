# %% [markdown]
# ### PASO 1: Verificar versión de Python

import sys  # Acceder a la información de la versión de Python.
import os  # Manejo de rutas, archivos y operaciones del sistema.
import logging  # Configuración y uso de logs para monitorear la ejecución.

# Configurar la variable de entorno para desactivar la paralelización de tokenizadores y evitar advertencias
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Definir la versión requerida de Python
REQUIRED_VERSION = (3, 10, 12)
current_version = sys.version_info

# Validar la versión de Python
if (current_version.major, current_version.minor, current_version.micro) != REQUIRED_VERSION:
    logging.warning("""
    **********************************************
    ** Advertencia: Versión de Python no compatible **
    **********************************************
    Este chatbot está optimizado para Python 3.10.12.
    La versión actual es Python {}.{}.{}.
    Algunas funcionalidades pueden no funcionar correctamente.
    **********************************************
    """.format(current_version.major, current_version.minor, current_version.micro))
else:
    logging.info("""
    **********************************************
    ** Versión de Python compatible **
    **********************************************
    Python 3.10.12 detectado correctamente.
    Todas las funcionalidades deberían operar sin problemas.
    **********************************************
    """)

# %% [markdown]
# ### PASO 2: Instalación de Paquetes Necesarios
# Antes de ejecutar esta aplicación, asegúrate de instalar todas las dependencias necesarias.
# Ejecuta el siguiente comando en tu terminal:
#
#     pip install -r requirements.txt
#
# Nota: Asegúrate de tener un archivo `requirements.txt` en el mismo directorio que este script.

# %% [markdown]
# ### PASO 3: Importar Librerías y Configurar Logging

# Importación de librerías esenciales para el funcionamiento del chatbot.
import os  # Manejo de rutas, archivos y operaciones del sistema.
import json  # Manipulación de datos en formato JSON.
import logging  # Configuración y uso de logs para monitorear la ejecución.
import hnswlib  # Búsqueda eficiente de similitud utilizando índices de alta dimensionalidad.
from sentence_transformers import SentenceTransformer, util  # Embeddings de texto y cálculo de similitud.
import numpy as np  # Operaciones matemáticas avanzadas y estructuras de datos.
from dotenv import load_dotenv  # Carga de variables de entorno desde un archivo `.env`.
from PyPDF2 import PdfReader  # Extracción de texto de documentos PDF.
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type  # Gestión de reintentos en funciones críticas.
from llama_index.llms.gemini import Gemini  # Interfaz para el modelo de lenguaje Gemini.
from llama_index.core.llms import ChatMessage  # Estructuras de mensajes para interacción con LLMs.
import time  # Manejo de tiempos y medición de duración de procesos.
import hashlib  # Generación de hashes únicos para almacenamiento en caché.
import random  # Generación de valores aleatorios, útil para respuestas personalizadas.
import unicodedata  # Normaliza texto de preguntas
import streamlit as st  # Importamos Streamlit

logging.info("Librerías importadas correctamente.")  # Log para confirmar la importación exitosa de todas las librerías.

# Carga de variables de entorno desde un archivo .env para proteger información sensible como claves de API.
load_dotenv()
logging.info("Variables de entorno cargadas correctamente.")

# Definición de funciones y clases globales que no dependen de Streamlit
def normalizar_texto(texto):
    """
    Normaliza el texto convirtiéndolo a minúsculas y eliminando espacios extra.
    """
    texto = texto.lower().strip()
    return texto

# Clase para representar documentos cargados con contenido y metadatos.
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text  # Contenido principal del documento.
        self.metadata = metadata or {}  # Metadatos adicionales, si existen, o un diccionario vacío.
    
    def __str__(self):
        """
        Representación en formato legible de los metadatos del documento.
        Accede a los metadatos de forma segura utilizando `.get()` para evitar errores si faltan claves.
        """
        return (
            f"Título: {self.metadata.get('Title', 'N/A')}\n"
            f"Resumen: {self.metadata.get('Summary', 'N/A')}\n"
            f"Tipo de Estudio: {self.metadata.get('StudyType', 'N/A')}\n"
            f"Paises donde se desarrolla el estudio: {self.metadata.get('Countries', 'N/A')}\n"
            f"Fase en que se encuentra el estudio: {self.metadata.get('Phases', 'N/A')}\n"
            f"Identificación en ClinicalTrial: {self.metadata.get('IDestudio', 'N/A')}.\n\n"
        )

# Clase para manejar índices HNSWlib y realizar búsquedas eficientes de similitud.
class HNSWIndex:
    def __init__(self, embeddings, metadata=None, space='cosine', ef_construction=200, M=16):
        self.dimension = embeddings.shape[1]  # Dimensión de los embeddings.
        self.index = hnswlib.Index(space=space, dim=self.dimension)  # Inicializa el índice HNSW con métrica de coseno.
        self.index.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=M)
        self.index.add_items(embeddings, np.arange(embeddings.shape[0]))
        self.index.set_ef(50)
        self.metadata = metadata or []
    
    def similarity_search(self, query_vector, k=5):
        """
        Realiza una búsqueda de los `k` elementos más similares al vector de consulta.
        Retorna una lista de tuplas con metadatos y distancias.
        """
        labels, distances = self.index.knn_query(query_vector, k=k)
        return [(self.metadata[i], distances[0][j]) for j, i in enumerate(labels[0])]

# Definición de funciones que no dependen del contexto de Streamlit
def generar_hash(pregunta):
    """
    Genera un hash SHA-256 único para la pregunta proporcionada.
    Este hash se utiliza como identificador para el almacenamiento en caché.
    """
    return hashlib.sha256(pregunta.encode('utf-8')).hexdigest()  # Convierte la pregunta a un hash hexadecimal.

def es_saludo(pregunta):
    """
    Detecta si la entrada del usuario es un saludo.
    """
    saludos = ["hola", "buen día", "buenas", "cómo estás", "cómo te llamas", "qué tal", "estás bien", "buenas tardes", "buenas noches"]
    return any(saludo in pregunta.lower() for saludo in saludos)

def responder_saludo():
    """
    Devuelve una respuesta aleatoria de saludo.
    """
    saludos_respuestas = [
        "¡Hola! Estoy para ayudarte con información sobre ensayos clínicos. ¿En qué puedo asistirte hoy?",
        "¡Buenas! ¿Tenés alguna pregunta sobre ensayos clínicos en enfermedades neuromusculares?",
        "¡Hola! ¿Cómo puedo ayudarte con tus consultas sobre ensayos clínicos?"
    ]
    return random.choice(saludos_respuestas)

# %% [markdown]
# ### PASO 4: Definir Funciones para Cargar Documentos

def extract_content(filepath):
    """
    Extrae el contenido del archivo según su tipo (.txt, .json, .pdf).
    """
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            units = content.split("\n-----\n")
            return units
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        elif filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            return ''.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        logging.error(f"Error al extraer contenido de '{filepath}': {e}")
        return None

def load_documents(source, is_directory=False):
    """
    Carga documentos desde un archivo o un directorio. Soporta .txt, .json, .pdf.
    """
    loaded_files = []
    if is_directory:
        for filename in os.listdir(source):
            filepath = os.path.join(source, filename)
            if os.path.isfile(filepath) and filepath.endswith(('.txt', '.json', '.pdf')):
                content = extract_content(filepath)
                if content:
                    loaded_files.append({"filename": normalizar_texto(filename), "content": content})
    else:
        content = extract_content(source)
        if content:
            loaded_files.append({"filename": normalizar_texto(os.path.basename(source)), "content": content})
    return loaded_files

# %% [markdown]
# ### PASO 5: Definir Función para Configurar Gemini

def configure_gemini():
    """
    Configura la instancia de Gemini utilizando la clave API almacenada en variables de entorno.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("La clave API de Gemini no está configurada.")
        st.error("Configura GEMINI_API_KEY en tu archivo .env.")
        st.stop()
    gemini_llm = Gemini(api_key=api_key)
    logging.info("Gemini configurado correctamente.")
    return gemini_llm

# %% [markdown]
# ### PASO 6: Definir Funciones para Configurar el Modelo de Embeddings y `doc_enfermedad`

def load_embedding_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Carga y cachea el modelo de embeddings."""
    return SentenceTransformer(model_name)

def precompute_doc_embeddings(documentos, model):
    """Precomputa y cachea los embeddings de los nombres de los documentos."""
    doc_filenames = [doc['filename'] for doc in documentos]
    doc_filenames_embeddings = model.encode(doc_filenames, show_progress_bar=True)
    logging.info("Embeddings de nombres de archivos precomputados.")
    return doc_filenames_embeddings

def generate_embedding(texto, model):
    """
    Genera un embedding para una pregunta o texto sin normalizar excesivamente el contenido.
    """
    try:
        embedding = model.encode([texto])
        logging.info("Embedding generado para el texto.")
        return embedding
    except Exception as e:
        logging.error(f"Error al generar el embedding: {e}")
        return None

def doc_enfermedad(pregunta_normalizada, doc_filenames_embeddings, model):
    """
    Identifica el índice del documento más relevante para la pregunta.
    """
    try:
        preg_embedding = generate_embedding(pregunta_normalizada, model)
        similarities = [util.cos_sim(preg_embedding, doc_emb).item() for doc_emb in doc_filenames_embeddings]
        max_index = similarities.index(max(similarities))
        logging.debug(f"Índice máximo: {max_index} con similitud {similarities[max_index]}")
        return max_index
    except Exception as e:
        logging.error(f"Error en doc_enfermedad: {e}")
        return None

# %% [markdown]
# ### PASO 7: Definir Funciones para Procesar Documentos y Crear Índices

def desdobla_doc(data2, model):
    """
    Convierte el contenido de los datos proporcionados en instancias de Document y crea un índice HNSWlib.
    """
    documents = []  # Lista para almacenar los objetos Document.
    summaries = []  # Lista para almacenar los resúmenes generados.

    for entry in data2['content']:  # Itera sobre cada entrada en los datos.
        # Extrae información relevante del contenido, manejando valores faltantes con cadenas vacías.
        nctId = entry.get("IDestudio", "")
        briefTitle = entry.get("Title", "")
        summary = entry.get("Summary", "")
        studyType = entry.get("StudyType", "")
        country = entry.get("Countries", "")
        overallStatus = entry.get("OverallStatus", "")
        conditions = entry.get("Conditions", "")
        phases = entry.get("Phases", "") or entry.get("Phasess", "")

        # Genera un resumen del estudio utilizando los datos extraídos.
        Summary = (
            f"The study titled '{briefTitle}', of type '{studyType}', "
            f"is being conducted to investigate the condition(s) {conditions}. "
            f"This study is briefly summarized as follows: {summary}. "
            f"Currently, the study status is {overallStatus}, and it is taking place in {country}. "
            f"The study is classified under {phases} phase. "
            f"For more information, search {nctId} on ClinicalTrials."
        )

        # Asocia el resumen con sus metadatos.
        metadata = {"Summary": Summary}

        # Crea una instancia de Document y la agrega a la lista.
        documents.append(Document(Summary, metadata))
        summaries.append(Summary)  # Almacena el resumen para posibles referencias.

    # Genera embeddings para todos los documentos utilizando el modelo cargado.
    embeddings = model.encode([doc.page_content for doc in documents], show_progress_bar=True)
    embeddings = np.array(embeddings).astype(np.float32)  # Convierte los embeddings a float32 para compatibilidad.

    # Crea un índice HNSWlib utilizando los embeddings generados.
    vector_store = HNSWIndex(embeddings, metadata=[doc.metadata for doc in documents])

    return documents, vector_store  # Retorna los documentos procesados y el índice creado.

def preparar_indices(documentos, model):
    """
    Genera trozos e índices para las enfermedades y los almacena en listas.
    """
    trozos_archivos = []  # Lista para almacenar los bloques procesados de documentos.
    index_archivos = []  # Lista para almacenar los índices HNSWlib.

    for doc in documentos:  # Itera sobre cada conjunto de documentos.
        trozos, index = desdobla_doc(doc, model)  # Procesa y crea el índice para cada conjunto.
        trozos_archivos.append(trozos)  # Almacena los bloques procesados.
        index_archivos.append(index)  # Almacena el índice creado.

    logging.info("Índices HNSWlib creados para todos los documentos.")  # Log para confirmar la creación exitosa de los índices.
    return trozos_archivos, index_archivos

# %% [markdown]
# ### PASO 8: Definir Funciones para Traducción y Generación de Respuestas

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
def traducir(texto, idioma_destino, gemini_llm):
    """
    Traduce texto al idioma especificado utilizando el modelo Gemini.
    """
    start_time = time.time()  # Inicia el contador para medir el tiempo de traducción.

    # Prepara los mensajes para enviar al modelo de lenguaje.
    mensajes = [
        ChatMessage(role="system", content="Actúa como un traductor."),
        ChatMessage(role="user", content=f"Por favor, traduce este texto al {idioma_destino}: {texto}")
    ]
    try:
        respuesta = gemini_llm.chat(mensajes)
        elapsed_time = time.time() - start_time  # Calcula el tiempo transcurrido.
        logging.info(f"Traducción completada en {elapsed_time:.2f} segundos.")
        return respuesta.message.content.strip()  # Devuelve la traducción limpia.
    except Exception as e:
        logging.error(f"Error al traducir: {e}")  # Registra el error en los logs.
        raise  # Vuelve a lanzar la excepción para que el decorador @retry pueda manejarla.

def obtener_contexto(pregunta, index, trozos, model, top_k=50):
    """
    Recupera los trozos de texto más relevantes para responder la pregunta.
    """
    try:
        # No traducir la pregunta al inglés antes de generar el embedding
        pregunta_normalizada = normalizar_texto(pregunta)
        pregunta_emb = generate_embedding(pregunta_normalizada, model)
        if pregunta_emb is None:
            logging.warning("No se pudo generar el embedding de la pregunta.")
            return "No se pudo procesar la pregunta debido a un error al generar el embedding."
    
        # Buscar en el índice
        results = index.similarity_search(pregunta_emb, k=top_k)
        if not results:
            logging.warning("No se encontró contexto relevante.")
            return "No se encontró información relevante para esta pregunta."
    
        texto = "\n".join([entry[0]["Summary"] for entry in results])
        return texto
    except Exception as e:
        logging.error(f"Error al obtener el contexto: {e}")
        return "Hubo un problema al recuperar la información. Por favor, intenta con otra pregunta."

def categorizar_pregunta(pregunta_en_ingles):
    """
    Clasifica la pregunta en categorías específicas en inglés.
    """
    categorias = {
        "treatment": ["treatment", "medication", "cure", "therapy", "drug", "intervention", "medications", "therapies"],
        "trial": ["trial", "trials", "study", "studies", "test", "research", "clinical trial", "clinical trials"],
        "criteria": ["criteria", "inclusion", "exclusion", "participants", "eligibility"],
        "result": ["result", "results", "effectiveness", "outcome", "outcomes", "success", "failure"],
        "location": ["city", "cities", "country", "countries", "location", "locations", "place", "places"],
        "prevention": ["prevention", "prevent", "avoiding", "avoid", "risk reduction", "reduce risk"],
        "duration": ["duration", "years", "months", "timeframe", "period", "length"],
    }
    for categoria, palabras in categorias.items():
        if any(palabra in pregunta_en_ingles.lower() for palabra in palabras):
            return categoria
    return "general"

def generar_prompt(categoria, pregunta_en_ingles):
    """
    Genera un prompt específico basado en la categoría de la pregunta en inglés.
    """
    prompts = {
        "treatment": f"Provide detailed information about treatments related to: {pregunta_en_ingles}.",
        "trial": f"Describe current clinical trials related to: {pregunta_en_ingles}.",
        "criteria": f"Explain inclusion and exclusion criteria for clinical trials on: {pregunta_en_ingles}.",
        "result": f"Explain the most recent results of clinical trials on: {pregunta_en_ingles}.",
        "location": f"Indicate the geographical locations where clinical trials are being conducted for: {pregunta_en_ingles}.",
        "prevention": f"Offer prevention strategies for: {pregunta_en_ingles}.",
        "duration": f"Describe the typical duration of clinical trials on: {pregunta_en_ingles}.",
    }
    return prompts.get(categoria, f"Please answer the following question about clinical trials: {pregunta_en_ingles}")

def generar_respuesta(pregunta_en_ingles, contexto, prompt_especifico, gemini_llm):
    """
    Genera una respuesta usando el contexto proporcionado y un prompt específico.
    """
    start_time = time.time()

    mensajes = [
        ChatMessage(role="system", content="You are a medical expert."),
        ChatMessage(role="user", content=f"{prompt_especifico}\nContext: {contexto}\nQuestion: {pregunta_en_ingles}")
    ]
    try:
        respuesta = gemini_llm.chat(mensajes)
        elapsed_time = time.time() - start_time
        logging.info(f"Respuesta generada en inglés en {elapsed_time:.2f} segundos.")
        return respuesta.message.content.strip()
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {e}")
        return "I'm sorry, there was an error generating the response."

def obtener_respuesta_cacheada(pregunta):
    """
    Recupera una respuesta previamente generada desde el caché, si existe.
    """
    hash_pregunta = generar_hash(pregunta)  # Genera el hash único para la pregunta.
    archivo_cache = f"cache/{hash_pregunta}.json"  # Define la ruta del archivo de caché.
    if os.path.exists(archivo_cache):  # Verifica si el archivo de caché existe.
        try:
            with open(archivo_cache, "r", encoding='utf-8') as f:
                datos = json.load(f)
                return datos.get("respuesta", None)
        except Exception as e:
            logging.error(f"Error al leer el caché para la pregunta '{pregunta}': {e}")
            return None
    return None

def guardar_respuesta_cacheada(pregunta, respuesta):
    """
    Almacena una respuesta en el caché para consultas futuras.
    """
    hash_pregunta = generar_hash(pregunta)  # Genera el hash único para la pregunta.
    archivo_cache = f"cache/{hash_pregunta}.json"  # Define la ruta del archivo de caché.
    try:
        os.makedirs(os.path.dirname(archivo_cache), exist_ok=True)  # Asegura que el directorio del caché exista.
        with open(archivo_cache, "w", encoding='utf-8') as f:  # Abre el archivo en modo escritura.
            json.dump({"pregunta": pregunta, "respuesta": respuesta}, f, ensure_ascii=False, indent=4)
            # Guarda la pregunta y la respuesta en formato JSON.

        logging.info(f"Respuesta cacheada para la pregunta: '{pregunta}'")  # Log de éxito al guardar en caché.
    except Exception as e:  # Manejo de errores al guardar en caché.
        logging.error(f"Error al guardar la respuesta en caché para la pregunta '{pregunta}': {e}")

def responder_pregunta(pregunta, index, trozos, model, gemini_llm):
    """
    Responde una pregunta del usuario integrando:
    - Búsqueda en caché.
    - Traducción y recuperación de contexto.
    - Generación de respuestas personalizadas.
    
    Incluye manejo de caché para optimizar el tiempo de respuesta.
    """
    try:
        # Verificar si la respuesta ya está en el caché.
        respuesta_cacheada = obtener_respuesta_cacheada(pregunta)
        if respuesta_cacheada:
            logging.info(f"Respuesta obtenida del caché para la pregunta: '{pregunta}'")
            return respuesta_cacheada

        # Traducir la pregunta al inglés para la generación de la respuesta
        pregunta_en_ingles = traducir(pregunta, "inglés", gemini_llm)
        if not pregunta_en_ingles or len(pregunta_en_ingles) < 5:
            logging.warning("Traducción de la pregunta fallida o insuficiente.")
            respuesta = "No se pudo procesar tu pregunta debido a un error en la traducción."
            return respuesta

        # Categorizar la pregunta para generar prompts específicos.
        categoria = categorizar_pregunta(pregunta_en_ingles)
        logging.info(f"Categoría de la pregunta: {categoria}")

        # Generar un prompt basado en la categoría.
        prompt_especifico = generar_prompt(categoria, pregunta_en_ingles)
        logging.info(f"Prompt específico generado: {prompt_especifico}")

        # Obtener el contexto relevante para la pregunta.
        contexto = obtener_contexto(pregunta, index, trozos, model)
        logging.debug(f"Contexto recuperado: {contexto[:200] if contexto else 'Sin contexto'}")
        if not contexto:
            logging.warning("No se encontró un contexto relevante para la pregunta.")
            respuesta = "No pude encontrar información relevante para responder tu pregunta."
            guardar_respuesta_cacheada(pregunta, respuesta)
            return respuesta

        # Generar la respuesta final utilizando el contexto y el prompt.
        respuesta = generar_respuesta(pregunta_en_ingles, contexto, prompt_especifico, gemini_llm)
        logging.debug(f"Respuesta generada: {respuesta[:200] if respuesta else 'Sin respuesta'}")

        try:
            # Traducir la respuesta al español
            respuesta_en_espanol = traducir(respuesta, "español", gemini_llm)
            logging.info(f"Respuesta traducida al español: {respuesta_en_espanol}")
        except Exception as e:
            logging.error(f"Error al traducir la respuesta: {e}")
            respuesta_en_espanol = "Lo siento, ocurrió un error al traducir la respuesta. A continuación, la respuesta en inglés:\n\n" + respuesta

        # Guardar la respuesta generada en el caché.
        guardar_respuesta_cacheada(pregunta, respuesta_en_espanol)

        return respuesta_en_espanol

    except Exception as e:
        logging.error(f"Error en el proceso de responder pregunta: {e}")
        return "Ocurrió un error al procesar tu pregunta."

# %% [markdown]
# ### PASO 9: Definir la Función Principal `main()` y Ejecutar la Aplicación

def main():
    # Asegurar que el directorio de caché exista
    os.makedirs("cache", exist_ok=True)

    # PASO 4: Cargar Documentos
    ruta_fuente = 'data'  # Define la ruta del directorio de documentos.
    documentos = load_documents(ruta_fuente, is_directory=True)  # Carga los documentos.
    logging.info(f"Se cargaron {len(documentos)} documentos exitosamente.")  # Log con el número total de documentos cargados.

    # PASO 5: Configurar Gemini
    gemini_llm = configure_gemini()  # Configurar Gemini.

    # PASO 6: Configurar el Modelo de Embeddings y Definir `doc_enfermedad`
    model = load_embedding_model()
    doc_filenames_embeddings = precompute_doc_embeddings(documentos, model)

    # PASO 7: Procesar Documentos y Crear Índices
    trozos_archivos, index_archivos = preparar_indices(documentos, model)

    # Mensaje de bienvenida e instrucciones para el usuario
    st.title("Chatbot de Ensayos Clínicos")
    st.write("""
        Conversemos sobre Ensayos Clínicos relacionados con las siguientes enfermedades neuromusculares:
        - Distrofia Muscular de Duchenne o de Becker
        - Enfermedad de Pompe
        - Distrofia Miotónica
        - Enfermedad de almacenamiento de glucógeno
             
        Por favor, escribí tu pregunta indicando claramente la enfermedad sobre la que deseas información.
    """)

    pregunta = st.text_input("Tu pregunta:")
    if st.button("Enviar") or pregunta:
        if pregunta.lower() in ['salir', 'chau', 'exit', 'quit']:
            st.write("¡Chau!")
            logging.info("El usuario ha finalizado la sesión.")
            st.stop()

        if es_saludo(pregunta):
            respuesta_saludo = responder_saludo()
            st.write(respuesta_saludo)
            logging.info("Se detectó un saludo del usuario.")
        else:
            pregunta_normalizada = normalizar_texto(pregunta)
            logging.info(f"Pregunta procesada: {pregunta_normalizada}")
            logging.debug(f"Pregunta original: {pregunta}")

            idn = doc_enfermedad(pregunta_normalizada, doc_filenames_embeddings, model)
            if idn is None:
                logging.warning(f"No se pudo identificar la enfermedad para la pregunta: '{pregunta}'")
                respuesta = "Lo siento, no pude identificar la enfermedad relacionada con tu pregunta."
                st.write(f"Respuesta: {respuesta}")
            else:
                try:
                    index = index_archivos[idn]
                    trozos = trozos_archivos[idn]

                    with st.spinner("Generando respuesta..."):
                        respuesta = responder_pregunta(pregunta_normalizada, index, trozos, model, gemini_llm)
                    st.write(f"Respuesta: {respuesta}")
                except Exception as e:
                    logging.error(f"Error al generar la respuesta para la pregunta '{pregunta}': {e}")
                    respuesta = "Lo siento, ocurrió un error al generar la respuesta."
                    st.write(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
