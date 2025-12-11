# 🕵️‍♂️ Análisis de Dígitos para OCR - Herramienta Interactiva

## 🎯 Objetivo Práctico
Esta aplicación simula un entorno real en una empresa de **OCR (Reconocimiento Óptico de Caracteres)**. El objetivo es ayudar al equipo (y al dueño, Juanpis) a resolver un problema crítico: **el sistema confunde números escritos a mano que se parecen**, como el "1" con el "7" o el "5" con el "6".

La herramienta permite visualizar miles de números en un mapa interactivo para descubrir qué técnica de Inteligencia Artificial logra separarlos mejor en "islas" distintas, evitando errores en contabilidad o seguridad.

## 🎮 Guía de Funcionalidades

### 1. Visualización con PCA (La foto borrosa)
*   **Qué probar:** Haz clic en "Ejecutar PCA".
*   **Lo que verás:** Una nube de puntos mezclados.
*   **Lección práctica:** Nos muestra que las técnicas simples son rápidas, pero "aplastan" la información, haciendo imposible distinguir números complejos.

### 2. Laboratorio UMAP (El organizador experto)
*   **Qué probar:** Haz clic en "Ejecutar UMAP" y juega con los controles.
    *   **Vecinos:** Ajusta para ver el panorama general o los detalles finos.
    *   **Dimensiones:** ¡Cambia a 3D para rotar el gráfico!
*   **Lo que verás:** Grupos de números (colores) bien separados.
*   **Lección práctica:** Es la herramienta más efectiva para este negocio. Logra separar los dígitos confusos, lo que mejorará la precisión del OCR.

### 3. Comparativa con t-SNE
*   **Qué probar:** Ejecuta t-SNE para compararlo con UMAP.
*   **Lección práctica:** Aunque también separa bien los grupos, verás que es más lento. Útil para validación, pero quizás no para producción masiva.

### 4. Sección Educativa
La app incluye explicaciones sencillas mediante analogías:
*   **PCA:** Como tomar una foto (se pierde profundidad).
*   **t-SNE/UMAP:** Como organizar invitados en una fiesta según sus amistades.

### 5. Rendimiento
Al final de la app, una gráfica compara los tiempos de ejecución, ayudando a tomar decisiones de costo/beneficio para la empresa.

## 🚀 Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando en tu terminal:

```bash
streamlit run app.py
```

O ir a [streamlit cloud](https://labtsne.streamlit.app/)