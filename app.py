import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
 
#st.markdown(
#"""
#<style>
#    header {visibility: hidden;}
#    .block-container {
#            margin-top: -4rem; /* Puedes ajustar el valor según necesites */
#        }
#</style>
#""",unsafe_allow_html=True
#)

st.markdown("""
# **Prueba de ingreso a OCR**

Eres contratado/a como **científico/a de datos** por una empresa especializada en **reconocimiento óptico de caracteres (OCR)**, propiedad de Juanpis, para mejorar su sistema de digitalización de documentos manuscritos. Aunque el modelo de OCR actual ha sido optimizado con grandes volúmenes de datos, aún enfrenta **problemas en la diferenciación de ciertos dígitos escritos a mano**, especialmente aquellos con trazos similares, como **"1" y "7"**, **"3" y "8"**, o **"5" y "6"**.

Estos errores de clasificación pueden generar problemas significativos en la conversión de documentos físicos a texto digital, impactando áreas clave como:
- **Contabilidad y Finanzas:** Errores en el reconocimiento de montos numéricos pueden causar discrepancias en balances financieros.
- **Seguridad y Autenticación:** Fallos en la lectura de códigos de seguridad o contraseñas pueden comprometer sistemas de acceso.
- **Automatización de Documentos:** Formularios mal digitalizados pueden llevar a registros erróneos en bases de datos oficiales.

Consciente de la importancia de mejorar la precisión del sistema, decides realizar un análisis para visualizar **cómo se distribuyen los dígitos manuscritos en un espacio de menor dimensión**. Esta exploración permitirá a la empresa tomar decisiones informadas sobre qué enfoques mejorarían el rendimiento del OCR. Para ello, propones comparar tres técnicas de reducción de dimensionalidad:

- **PCA (Análisis de Componentes Principales):** Un método lineal que encuentra las direcciones de mayor varianza en los datos, útil para visualizar estructuras globales.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Un enfoque no lineal que preserva relaciones locales y es efectivo para descubrir agrupaciones naturales.
- **UMAP (Uniform Manifold Approximation and Projection):** Una técnica más reciente que combina ventajas de PCA y t-SNE, preservando tanto la estructura local como global.

Tu objetivo es determinar **qué técnica de reducción de dimensionalidad permite visualizar mejor la estructura de los dígitos y su separación**, ayudando a identificar patrones que puedan mejorar la precisión del OCR en escenarios reales.

            
# **Preguntas**
#### 1. ¿Cómo se diferencian los resultados de PCA, t-SNE y UMAP en la representación de los dígitos?
""")

if "X_subset" not in st.session_state:
    data = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)

    # Muestreo de datos
    X_subset, y_subset = shuffle(X, y, random_state=42, n_samples=15000)

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Guardar en session_state
    st.session_state.X_subset = X_subset
    st.session_state.y_subset = y_subset
    st.session_state.X_scaled = X_scaled

# Recuperar datos desde session_state
X_subset = st.session_state.X_subset
y_subset = st.session_state.y_subset
X_scaled = st.session_state.X_scaled

#umap_reducer = umap.UMAP(n_components=2, random_state=42,min_dist=0.005,n_neighbors=10)
#X_umap = umap_reducer.fit_transform(X_scaled)

#tsne = TSNE(n_components=2, random_state=42, perplexity=10)
#X_tsne = tsne.fit_transform(X_subset)

if "pca_fig" not in st.session_state:
    st.session_state.pca_fig = None

if "umap_fig" not in st.session_state:
    st.session_state.umap_fig = None

if "tsne_fig" not in st.session_state:
    st.session_state.tsne_fig = None

# UI para PCA
col1, col2 = st.columns([0.9, 0.1])

with col1:
    st.markdown("##### Proyección de MNIST con PCA")

with col2:
    ejecutar_pca = st.button("Ejecutar PCA")

if ejecutar_pca:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig_pca, ax_pca = plt.subplots(figsize=(16, 6))
    scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset, cmap='tab10', alpha=0.6, s=5)
    ax_pca.set_xlabel("Componente 1")
    ax_pca.set_ylabel("Componente 2")
    plt.colorbar(scatter, label="Dígitos")

    st.session_state.pca_fig = fig_pca  # Guardar figura en memoria

# Mostrar la última gráfica guardada de PCA si existe
if st.session_state.pca_fig:
    st.pyplot(st.session_state.pca_fig)

# UI para UMAP
col1, col2 = st.columns([0.9, 0.1])

with col1:
    st.markdown("##### Proyección de MNIST con UMAP")

with col2:
    ejecutar_umap = st.button("Ejecutar UMAP")

# Parámetros de UMAP
n_neighbors_list = [5, 10, 15, 20, 30, 50]
n_components_list = [2, 3]
min_dist_list = [0.001, 0.01, 0.05, 0.1, 0.2]
spread_list = [0.1, 0.5, 0.8, 1.0, 1.5] 
metric_list = ["euclidean", "manhattan", "chebyshev", "cosine", "correlation"]  
repulsion_strength_list = [0.5, 1.0, 1.5, 2.0, 3.0]
data = ["Escalados", "Sin escalar"]

col1, col2, col3, col4, col5, col7, col8 = st.columns(7)

with col1:
    tip_data = st.selectbox("Tipo de datos:", data, 1)
    tipo_data = X_scaled if tip_data == "Escalados" else X_subset

with col2:
    n_neighbors = st.selectbox("Número de vecinos:", n_neighbors_list, 2)

with col3:
    min_dist = st.selectbox("Distancia mínima:", min_dist_list, 2)

with col4:
    spread = st.selectbox("Dispersión:", spread_list, 3)

with col5:
    metric = st.selectbox("Distancia:", metric_list)

with col7:
    repulsion = st.selectbox("Separación:", repulsion_strength_list,1)

with col8:
    n_components = st.selectbox("Dimensiones:", n_components_list)

if ejecutar_umap:
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,  # Mantener 2D para visualización
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        repulsion_strength=repulsion,
        random_state=42
    )

    embedding_umap = umap_model.fit_transform(tipo_data)

    if n_components == 2:
        # Visualización 2D con Matplotlib
        fig_umap, ax_umap = plt.subplots(figsize=(16, 6))
        scatter = ax_umap.scatter(embedding_umap[:, 0], embedding_umap[:, 1], s=2, alpha=0.6, c=y_subset, cmap='tab10')
        plt.colorbar(scatter, label="Dígitos")
        ax_umap.set_xticks([])
        ax_umap.set_yticks([])

        st.session_state.umap_fig = fig_umap

    elif n_components == 3:
        
        umap_df = pd.DataFrame(embedding_umap, columns=["UMAP1", "UMAP2", "UMAP3"])
        umap_df["Dígito"] = y_subset

        fig_umap = px.scatter_3d(
            umap_df, x="UMAP1", y="UMAP2", z="UMAP3", color=umap_df["Dígito"].astype(str),
            title=" ",
            opacity=0.7, size_max=5, color_discrete_sequence=px.colors.qualitative.Set3
        )

        st.session_state.umap_fig = fig_umap

if st.session_state.umap_fig:
    if isinstance(st.session_state.umap_fig, plt.Figure):
        st.pyplot(st.session_state.umap_fig)
    else:
        st.plotly_chart(st.session_state.umap_fig)

st.markdown(f"""
Peores:{"&emsp;"*2}Tipo de datos:Escalados{"&emsp;"*2}Número de vecinos:5{"&emsp;"*2}Distancia mínima:0.001{"&emsp;"*2}Dispersión:0.1{"&emsp;"*2}Distancia:euclidean{"&emsp;"*2}Separación:1.0{"&emsp;"*2}Dimensiones:2

""",True)

col1, col2 = st.columns([0.9, 0.1])

with col1:
    st.markdown("##### Proyección de MNIST con TSNE")

with col2:
    ejecutar_tsne = st.button("Ejecutar TSNE")

n_components_list = [2, 3]
perplexity_list = [5, 10, 20, 30, 40, 50, 100]  
early_exaggeration_list = [4.0, 8.0, 12.0, 20.0, 30.0]  
metric_list = ["euclidean", "manhattan", "chebyshev", "cosine", "correlation", "mahalanobis"]  
init_list = ["random", "pca", "spectral"]  
data = ["Escalados", "Sin escalar"]

col1, col2, col3, col5, col7, col8 = st.columns(6)

with col1:
    tip_data = st.selectbox("Tipo de datos: ", data, 1)
    tipo_data_tsne = X_scaled if tip_data == "Escalados" else X_subset

with col2:
    perplexity_tsne = st.selectbox("Número de vecinos: ", perplexity_list)

with col3:
    early_exaggeration_tsne = st.selectbox("Separación: ", early_exaggeration_list)

with col5:
    metric_tsne = st.selectbox("Distancia: ", metric_list)

with col7:
    init_tsne = st.selectbox("Inicializacion de puntos:", init_list)

with col8:
    n_components_tsne = st.selectbox("Dimensiones: ", n_components_list)

if ejecutar_tsne:

    tsne = TSNE(n_components=n_components_tsne, 
                random_state=42, 
                perplexity=perplexity_tsne, 
                init= init_tsne, 
                early_exaggeration= early_exaggeration_tsne, 
                metric= metric_tsne)
    
    X_tsne = tsne.fit_transform(tipo_data_tsne)

    if n_components_tsne == 2:
        fig_tsne, ax_tsne = plt.subplots(figsize=(16, 6))
        scatter = ax_tsne.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter, label="Dígitos")
        st.session_state.tsne_fig = fig_tsne 

    elif n_components_tsne == 3:

        tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2", "TSNE3"])
        tsne_df["Dígito"] = y_subset

        fig_tsne = px.scatter_3d(
            tsne_df, x="TSNE1", y="TSNE2", z="TSNE3", color=tsne_df["Dígito"].astype(str),
            title=" ",
            opacity=0.7, size_max=5, color_discrete_sequence=px.colors.qualitative.Set3
        )

        st.session_state.tsne_fig = fig_tsne

if st.session_state.tsne_fig:
    if isinstance(st.session_state.tsne_fig, plt.Figure):
        st.pyplot(st.session_state.tsne_fig)
    else:
        st.plotly_chart(st.session_state.tsne_fig)


st.markdown("#### 2. ¿Cuál de estas técnicas parece agrupar mejor los números similares? ¿Por qué?")

st.markdown("""
- **PCA:** los datos estan muy dispersos y no se observan grupos claramente definidos. Esto es por que PCA conserva la varianza global y linealiza los datos, 
por lo que no mantiene las estructuras ni distancias en los datos y tampoco captura las estructura no lineal de los datos

- **UMAP:** Se observan grupos bien separados, esta técnica al enfocarse en preservar la estructura y las distancias de los datos logra divir los bien los digitos

- **TSNE:** También muestra agrupaciones claras, aunque en comparación con UMAP, los grupos de digitos no estan concentrados todos en un mismo punto, para lograr una
separacion mejor puede requerir más ajuste en los hiperparámetros.

**UMAP** es la mejor opción, ya que forma grupos compactos y separados, logrando preservar las estructuras locales con grupos compactos, como la separacion entre grupos 
dejando que los grupos que estan mas cerca, sean los digitos que mas se parecen o mas caracteristicas tienen en comun al ser dibujados
            """)

st.markdown("#### 3. ¿Cómo podrías interpretar las agrupaciones formadas en cada método? ¿Existen dígitos que se solapan en ciertas técnicas?")

st.markdown("""
Sí, especialmente en **PCA**, los dígitos se mezclan en la misma región y esto susede por ser una tecnica para maximizar varianza y no preservar las estructuras en los datos. 
En UMAP y t-SNE, la separación es mucho mejor, pero aún existen casos donde algunos dígitos con características similares como lo son el 4 y el 9 o el 3 y el 8 se solapan en ciertas regiones.
""")

st.markdown("#### 4. ¿Qué impacto podría tener la selección de la técnica de reducción de dimensionalidad en la eficiencia del sistema OCR?")

st.markdown("""
La elección de la técnica de reducción de dimensionalidad puede influir significativamente en la precisión del sistema OCR, 
            especialmente en la diferenciación de dígitos manuscritos similares.
            
- **PCA:** Como conserva la varianza global pero no captura bien estructuras no lineales, no es la mejor opción para separar dígitos con trazos similares. 
            Esto puede aumentar los errores en la clasificación de caracteres como "1" y "7" o "5" y "6" que son numeros similares en su estructura 

- **UMAP y t-SNE:** Estas técnicas son mejores opciones para la separación de clases al preservar relaciones no lineales y las distancias en los datos.

Eficiencia en el procesamiento de documentos

- **PCA:** es computacionalmente más eficiente pero menos preciso en la separación de dígitos similares.
            
- **TSNE:** es preciso pero computacionalmente costoso, lo que podría ralentizar el proceso en producción y es mas sencible con sus hiperparametros para identificar
            que digitos son mas similares con otros.

- **UMAP:** ofrece un balance entre precisión y velocidad, precision en que forma grupos compactos y ademas permite visualizar que digitos suelen confudirse mas 
            con otros digitos, lo que permite tomar decisiones mas orientadas a los problemas en los datos y ademas esta tecnica es menos costosa computacionalmente
""")

st.markdown("#### 5. ¿Crees que la escala de los datos afecta los resultados de estas técnicas? ¿Cómo podrías comprobarlo?")

st.markdown("""
- **PCA**: Al maximiza la varianza a lo largo de las componentes principales. Si las características tienen escalas muy diferentes, aquellas con mayor mayor magnitud
            y mayor varianza sesgaran la transformación.

- **UMAP**: Es menos sensible a la escala que PCA, se basa en la distancia más corta entre dos puntos pero si una variable tiene valores mucho más grandes que otras, 
            puede influir en la distancia y sesgar la agrupación.

- **TSNE**: Calcula distancias entre puntos en un espacio de alta dimensión antes de proyectarlos a una menor dimensión. si las características tienen rangos muy 
            distintos, las distancias pueden sesgarse. 

Se puede comprobar mediante la experimentacion al ver el resultado de la separacion con los datos escalados y sin escalar, donde se esperaria que escalados den un
resultado mejor si las caracteristicas tienen rangos muy distintos
""")

st.markdown("#### 6. ¿Es necesario escalar los datos antes de aplicar PCA, t-SNE y UMAP? ¿Por qué o en qué casos sí o no?")

st.markdown("""
- **PCA**: Sí, es recomendable escalar los datos antes de PCA. Al Busca maximizar la varianza de los datos, y si las variables tienen escalas diferentes, 
            aquellas con mayor magnitud sesgaran la reducción de dimensionalidad

- **TSNE**: Sí, generalmente se recomienda escalar, al usar la distancias euclidianas para calcular similitudes, los valores en distintas escalas pueden 
            sesgar los resultados. 

- **UMAP**: Depende de la métrica usada, Si se usa la métrica euclidiana, es recomendable escalar, ya que las diferencias de escala afectan la forma en 
            que se calcula la distancia entre puntos, usas métricas como "cosine" o "correlation", no siempre es necesario escalar, porque estas métricas 
            son invariante a la escala.

""")

st.markdown("""#### 7. Juanpis tiene curiosidad sobre cómo funcionan PCA, t-SNE y UMAP. ¿Cómo le explicarías de forma corta y concisa las diferencias clave \
            entre estas técnicas?""")  

_, col1, col2 = st.columns([0.1,0.6,0.3])

with col1:
    st.image("vYsQF.png",width= 700)

with col2:
    st.markdown("""
    - **PCA**: Aqui esta su hijo juampis, y tiene un outifit muy particular, el le pide que le tome una foto, pero es una foto \
                donde no pierda ningun detalle de su outfit, entonces usted empieza a buscar en que angulo tomar la foto, y los \
                resultados son los de la parte derecha, como puede ver, en el angulo 1 y 3 no logro camputarar todo el outif y \
                perdio informacion o detalle del outif, mientras que en el angulo 2, logro caputrar todo el outfit por lo que no \
                perdio informacion y le dio contentillo a su hijo
    """)

st.markdown("""
- **TSNE**: Imaginese que quiere organizar una fiesta, pero en las personas invitadas existen grupos de amigos que no se pueden ver \
            ver con otros grupos por que son muy distintos, pues lastimosamente el organizador TSNE no tuvo esto en cuenta, fue un \
            excelente organizador y logro dejar a los grupos de amigos cerca unos de otros, pero olvido el pequeño detalle y no \
            necesariamente dejo a los grupos distintos lejos, puede ser que el azar dejara a los grupos distintos lejos, pero el \
            organizador al olvidar este detalle, no siempre el azar jugara a su favor y puede pasar que estos grupos muy distintos \
            queden cercanos

- **UMAP**: Ahora viendo el error pasado y que esto de tener a grupos muy distintos cerca fue problematico, decide cambiar de organizador \
            y contrata a UMAP, este a diferencia de TSNE, si tiene muy presente a quienes no tiene que sentar cerca y a la vez sabe quienes \
            son del mismo grupo, por lo que logra agrupar a los amigos en sus grupos usales, y alejarlos con los que tienen muchas diferencias

""")

_, col1, _ = st.columns([0.2,0.6,0.1])

with col1:
    st.image("evolution-embeddings.jpg",width= 700)


st.markdown("""#### 10. Como parte del proceso de validación de autoría, la empresa verifica que cada desarrollador comprenda a fondo el código, pudiendo \
            adaptarlo y modificarlo según los requerimientos del negocio. Para garantizar esto, el equipo de revisión técnica, liderado por Albert, realizará \
            una serie de preguntas y pruebas destinadas a evaluar tu dominio sobre el código, su propósito y su implementación. Deberás estar preparado para \
            justificar cada decisión tomada en el desarrollo""")


code = """
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,                 # Dimensiones de la reducción (2D o 3D)
    perplexity=30.0,                # Balance entre agrupamiento local y global
    early_exaggeration=12.0,        # Expande distancias en la fase inicial
    learning_rate='auto',           # Controla la velocidad de ajuste
    max_iter=None,                  # Iteraciones (None usa el valor predeterminado)
    n_iter_without_progress=300,    # Iteraciones sin mejora antes de detenerse
    min_grad_norm=1e-07,            # Criterio de convergencia basado en gradientes
    metric='euclidean',             # Distancia utilizada (Euclidiana por defecto)
    metric_params=None,             # Parámetros adicionales para la métrica
    init='pca',                     # Inicialización ('random' o 'pca')
    verbose=0,                      # Nivel de mensajes en la ejecución
    random_state=None               # Control de aleatoriedad para reproducibilidad
)

from umap import UMAP

umap_model = UMAP(
    n_neighbors=15,                 # Número de vecinos considerados en la reducción
    n_components=2,                 # Dimensiones de la reducción (2D o 3D)
    metric='euclidean',             # Distancia utilizada para medir similitud
    init='spectral',                # Método de inicialización ('random' o 'spectral')
    min_dist=0.1,                   # Distancia mínima entre puntos en el espacio reducido
    spread=1.0,                     # Controla la dispersión de los datos embebidos
    repulsion_strength=1.0,         # Fuerza de separación entre puntos cercanos
    random_state=None,              # Control de aleatoriedad para reproducibilidad
    target_metric='categorical',    # Distancia usada para datos supervisados
    verbose=False                   # Mensajes de estado en la ejecución
)
"""

st.code(code, language="python")



st.markdown("#### 11. ¿Cuál de estas técnicas es más adecuada para grandes volúmenes de datos y por qué?")

st.markdown("""Se realizara una prueba con distintos tamaños de muestra y varias pruebas con el mismo tamaño para evitar estar sesgados a una sola prueba
            y a la vez comparar los tiempos que necesita cada algoritmo para reducir la dimensionalidad en contraste con su calidad
            """)

st.code("""
sample_sizes = [1000, 3000, 5000, 7000, 10000, 12000, 15000]
n_iter = 5  # Número de iteraciones por cada tamaño

# Diccionario para almacenar tiempos
time_results = {"PCA": [], "t-SNE": [], "UMAP": []}

for n in sample_sizes:
    pca_times, tsne_times, umap_times = [], [], []
    
    for _ in range(n_iter):
        X_subset, _ = shuffle(X, y, random_state=None, n_samples=n)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)

        # PCA
        start = time.time()
        PCA(n_components=2).fit_transform(X_scaled)
        pca_times.append(time.time() - start)
        
        # t-SNE
        start = time.time()
        TSNE(n_components=2, perplexity=10).fit_transform(X_subset)
        tsne_times.append(time.time() - start)
        
        # UMAP
        start = time.time()
        umap.UMAP(n_components=2, min_dist=0.005,n_neighbors=10).fit_transform(X_subset)
        umap_times.append(time.time() - start)
    
    time_results["PCA"].append((np.mean(pca_times), np.std(pca_times)))
    time_results["t-SNE"].append((np.mean(tsne_times), np.std(tsne_times)))
    time_results["UMAP"].append((np.mean(umap_times), np.std(umap_times)))

df_results = pd.DataFrame({
    "Sample Size": sample_sizes,
    "PCA Mean": [x[0] for x in time_results["PCA"]],
    "PCA Std": [x[1] for x in time_results["PCA"]],
    "t-SNE Mean": [x[0] for x in time_results["t-SNE"]],
    "t-SNE Std": [x[1] for x in time_results["t-SNE"]],
    "UMAP Mean": [x[0] for x in time_results["UMAP"]],
    "UMAP Std": [x[1] for x in time_results["UMAP"]]
})

        """)

if "pca_tiempo" not in st.session_state:
    st.session_state.pca_tiempo = None

# Cargar datos solo si no están en memoria
if st.session_state.pca_tiempo is None:
    # Leer CSV
    tiempos = pd.read_csv("Tiempos.csv")

    # Lista de tamaños de muestra
    sample_sizes = tiempos["Muestras"].tolist() if "Muestras" in tiempos.columns else [1000, 3000, 5000, 7000, 10000, 12000, 15000]

    # Crear figura
    fig, ax = plt.subplots(figsize=(18, 6))

    for method, color in zip(["PCA", "t-SNE", "UMAP"], ["blue", "red", "green"]):
        mean_col = f"{method} Mean"
        std_col = f"{method} Std"
        
        if mean_col in tiempos.columns and std_col in tiempos.columns:
            ax.plot(sample_sizes, tiempos[mean_col], label=method, color=color, marker='o')
            ax.fill_between(sample_sizes, 
                            np.array(tiempos[mean_col]) - np.array(tiempos[std_col]), 
                            np.array(tiempos[mean_col]) + np.array(tiempos[std_col]), 
                            color=color, alpha=0.2)

    # Estilos del gráfico
    ax.set_xlabel("Número de muestras")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.legend()
    ax.set_title("Comparación de tiempo de ejecución: PCA vs t-SNE vs UMAP")

    # Guardar en memoria
    st.session_state.pca_tiempo = fig

# Mostrar la figura
st.pyplot(st.session_state.pca_tiempo)