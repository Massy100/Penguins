# Análisis y Clasificación de Pingüinos con Árbol de Decisión

### 1. Distribución de Especies de Pingüinos
![Distribución de especies de pingüinos](/img/grafica1.png)

**¿Qué significa?**  
Este gráfico de barras muestra cuántos pingüinos de cada especie existen en el dataset limpio (sin valores nulos).  
- Las especies son: **Adelie**, **Chinstrap** y **Gentoo**.  
- Podemos observar que la especie **Adelie** es la más abundante, seguida de Gentoo y finalmente Chinstrap.  
- Esto es importante porque si una especie tiene muy pocos ejemplares, el modelo podría tener más dificultad para aprender a clasificarla correctamente.

---

### 2. Relación entre Largo y Profundidad del Pico por Especie
![Relación entre largo y profundidad del pico](/img/grafica2.png)

**¿Qué significa?**  
Este gráfico de dispersión (scatter plot) relaciona dos características físicas: el largo del pico (eje X) y la profundidad del pico (eje Y).  
- Cada color representa una especie y cada forma representa el sexo del pingüino.  
- **Observación clave**: Las especies se separan naturalmente en regiones distintas del gráfico.  
  - Los pingüinos **Adelie** tienden a tener picos más cortos pero más profundos.  
  - Los **Gentoo** tienen picos más largos y menos profundos.  
  - Los **Chinstrap** se encuentran en un punto intermedio.  
- Esta separación sugiere que estas variables serán muy útiles para el modelo de clasificación.

---

### 3. Masa Corporal por Especie
![Masa corporal](/img/grafica3.png)

**¿Qué significa?**  
Aquí se presentan dos gráficos: un **boxplot** (izquierda) y un **violinplot** (derecha), ambos mostrando la distribución de la masa corporal para cada especie.  
- **Boxplot**: Muestra la mediana, los cuartiles y los valores atípicos.  
- **Violinplot**: Además de mostrar la mediana y los cuartiles, muestra la densidad de los datos (en qué valores se concentran más los pingüinos).  
- **Conclusión**: Los pingüinos **Gentoo** son claramente más pesados que las otras dos especies. Adelie y Chinstrap tienen masas corporales similares, aunque los Chinstrap tienden a ser ligeramente más pesados.

---

### 4. Matriz de Correlación
![Matriz de correlación](/img/grafica4.png)

**¿Qué significa?**  
Este mapa de calor (heatmap) muestra cómo se relacionan las variables numéricas entre sí.  
- Los valores van de **-1 a 1**:  
  - **Cercano a 1**: Correlación positiva fuerte (cuando una variable aumenta, la otra también).  
  - **Cercano a -1**: Correlación negativa fuerte (cuando una variable aumenta, la otra disminuye).  
  - **Cercano a 0**: Sin correlación.  
- **Hallazgos**:  
  - La longitud del pico y la longitud de la aleta tienen una correlación positiva alta (0.65).  
  - La masa corporal también está fuertemente correlacionada con la longitud de la aleta (0.87).  
  - La profundidad del pico tiene una correlación negativa con la longitud del pico (-0.23) y con la masa corporal (-0.47).  
- Esto nos ayuda a entender qué variables están relacionadas y podrían aportar información redundante al modelo.

---

### 5. Relaciones entre Variables por Especie (Pairplot)
![Relaciones entre variantes](/img/grafica5.png)

Este gráfico de matriz de dispersión muestra todas las combinaciones posibles entre las variables numéricas, coloreadas por especie.  
- Confirma visualmente que las especies son separables usando combinaciones de estas variables.  
- Por ejemplo, la relación entre **longitud de la aleta** y **masa corporal** muestra una separación casi perfecta de los pingüinos Gentoo.

---

## Clasificación con Árbol de Decisión

### 6. Importancia de las Características
![Importancia de las características](/img/grafica6.png)

**¿Qué significa?**  
Este gráfico de barras muestra qué tan importante fue cada variable para que el árbol de decisión realice sus clasificaciones.  
- **Las variables más importantes** son:  
  1. **Longitud de la aleta (flipper_length_mm)**  
  2. **Profundidad del pico (bill_depth_mm)**  
  3. **Masa corporal (body_mass_g)**  
- Esto tiene sentido biológico: los pingüinos Gentoo se distinguen claramente por su mayor tamaño y aletas más largas.  
- Las variables codificadas como **sexo** e **isla** tuvieron muy poca importancia, lo que indica que no son determinantes para predecir la especie.

---

### 7. Árbol de Decisión
![Árbol de decisión](/img/grafica7.png)

**¿Qué significa?**  
Esta es la representación visual del árbol de decisión entrenado.  
- Cada **nodo** representa una pregunta sobre una característica (ej. "¿la longitud de la aleta es menor o igual a 206.5 mm?").  
- Dependiendo de la respuesta, se sigue una rama hacia la izquierda o derecha hasta llegar a una **hoja** que indica la especie predicha.  
- El árbol es interpretable y nos permite entender cómo el modelo toma decisiones.  
- En este caso, vemos que la primera división (raíz) se basa en la **longitud de la aleta**, lo que confirma su importancia.

---

### 8. Matriz de Confusión
![Matriz de confusión](/img/grafica8.png)

**¿Qué significa?**  
Esta matriz compara las predicciones del modelo con los valores reales.  
- Las filas representan la especie **real**.  
- Las columnas representan la especie **predicha**.  
- Los valores en la diagonal principal son las **predicciones correctas**.  
- Los valores fuera de la diagonal son los **errores**.  
- **Resultado**: El modelo clasifica correctamente casi todos los pingüinos, con solo 1 error en la especie Chinstrap (fue clasificada como Adelie).  
- Esto indica un rendimiento excelente.

---

**Resultados**:  
- Las especies **Adelie** y **Gentoo** se clasifican perfectamente (precisión y recall de 1.00).  
- La especie **Chinstrap** tiene una precisión de 0.93 y recall de 1.00, lo que significa que todos los Chinstrap fueron identificados, pero uno de ellos fue clasificado erróneamente como Adelie.  
- **Exactitud global (accuracy)**: **0.98**, lo que significa que el 98% de las predicciones fueron correctas.



