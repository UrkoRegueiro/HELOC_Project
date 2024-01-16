# Línea de Crédito con Garantía Hipotecaria. Predicción de Riesgo.

<div align="center">

  <img src="https://daxg39y63pxwu.cloudfront.net/hackerday_banner/hq/loan-default-risk-prediction-machine-learning-project.jpg" alt="Loan Default Risk Prediction" width="80%">
  
</div>

## Tecnologías usadas

**Lenguaje:** Python.

**Librerias:** numpy, pandas, matplotlib, seaborn, joblib, tensorflow, keras, sklearn, xgboost

------------

<h2>
  
Para visualizar la versión detallada del presente proyecto véase [HELOC_Project](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/heloc_project.ipynb)

<h2>
  
------------

## 1. **Introducción**

Este proyecto se centra en un conjunto de datos anonimizados de solicitudes de líneas de crédito con garantía hipotecaria (HELOC) realizadas por propietarios reales. Un **HELOC** es una línea de crédito ofrecida típicamente por un banco como un porcentaje del capital acumulado en una vivienda (la diferencia entre el valor de mercado actual de una vivienda y su precio de compra). Los clientes en este conjunto de datos han solicitado una línea de crédito en el rango de `5,000$` a `150,000$`. La **<u>tarea fundamental</u>** de este proyecto es utilizar la información sobre el solicitante en su informe crediticio para **<u>predecir si pagará su cuenta de HELOC en un plazo de 2 años</u>**. Esta predicción se utiliza luego para decidir si el propietario califica para una línea de crédito y, en caso afirmativo, cuánto crédito se debe otorgar.

Antes de proseguir con nuestro estudio es importante definir los siguientes términos:
- <u>**"Cuenta Comercial"**</u>:<br>Las cuentas comerciales de un cliente se refieren a las relaciones financieras que el cliente ha establecido con diversas entidades comerciales. Estas cuentas representan acuerdos o transacciones financieras que el cliente mantiene con empresas, instituciones financieras o proveedores de servicios. Las cuentas comerciales pueden incluir una variedad de productos y servicios financieros, y las transacciones asociadas con estas cuentas quedan registradas en el historial crediticio del cliente. La información de las cuentas comerciales que posee un cliente nos da una idea del comportamiento financiero de este.
    
- <u>**"Préstamos de Instalación"**</u>:<br> Estos son préstamos que se otorgan para financiar la compra de bienes duraderos o servicios específicos que generalmente se pagan en cuotas fijas a lo largo del tiempo. Estos préstamos son comunes para la adquisición de activos como automóviles, electrodomésticos, muebles, mejoras en el hogar, entre otros.
    - <u>**"Carga de Préstamos de Instalación"**</u>:<br> Es la proporción de la deuda relacionada con estos préstamos específicos con respecto al límite total de crédito disponible. Este indicador se utiliza para comprender la diversidad de las deudas de un individuo y evaluar cómo están utilizando su crédito para diferentes propósitos. Un bajo porcentaje puede indicar una gestión más equilibrada y diversa del crédito.<br>
    *<u>Ejemplo</u>*: Supongamos que tenemos una tarjeta de crédito con un límite total de `10,000$`. De ese límite, utilizamos `2,000$` para financiar la compra de muebles nuevos para el hogar, que pagamos en cuotas mensuales. La carga de préstamo de instalación sería del 20%, lo que significa que el 20% de tu límite total de crédito se está utilizando específicamente para préstamos de instalación.<br><br>
    
- <u>**"Consulta crediticia"**</u>:<br> Es la revisión de la información crediticia de un individuo por parte de una entidad financiera u otra institución autorizada. Esta consulta se realiza para evaluar la solvencia crediticia de la persona.
    
- <u>**"Línea de Crédito Rotativo"**</u>:<br> Es una forma de crédito renovable que permite a los individuos tomar prestado repetidamente hasta un límite preestablecido. Las tarjetas de crédito son un ejemplo común de líneas de crédito rotativo.
    - <u>**"Carga de Crédito Rotativo"**</u>:<br> Representa la cantidad de la línea de crédito rotativo que está actualmente en uso. Se expresa típicamente como un porcentaje que indica cuánto del crédito disponible se está utilizando actualmente. La carga de crédito rotativo es un factor importante en la evaluación del riesgo crediticio ya que, un alto porcentaje de carga de crédito rotativo puede interpretarse como una señal de riesgo, indicando una mayor dependencia del crédito.<br>
    *<u>Ejemplo</u>*: Una persona tiene una tarjeta de crédito con un límite de `1,000$` y ha utilizado $300, la carga de crédito rotativo sería del 30%.

A continuación se muestra la información contenida en el informe crediticio de un cliente que, poteriormente, se tendrán en cuenta a la hora de predecir si este presenta riesgo de impago o no:

<div align="center">

| **Variable**                           | **Descripción**                                                                                    | **Utilidad**                                                                                                                                                       |
|----------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **'RiskPerformance'**                  | Variable a predecir que indica el rendimiento de riesgo del cliente.                               | Esencial para la predicción y evaluación final del riesgo crediticio.                                                                                                |
| **'ExternalRiskEstimate'**             | Estimación numérica del riesgo crediticio externo asociado al cliente.                              | Un valor más alto indica un mayor riesgo, crucial para evaluar la capacidad del cliente para cumplir con los pagos.                                                 |
| **'MSinceOldestTradeOpen'**            | Meses desde la apertura de la cuenta comercial más antigua del cliente.                              | La estabilidad a largo plazo puede sugerir un comportamiento crediticio más confiable.                                                                             |
| **'MSinceMostRecentTradeOpen'**        | Meses desde la apertura de la cuenta comercial más reciente del cliente.                             | Indica la antigüedad de la cuenta comercial más reciente en meses.                                                                                                   |
| **'AverageMInFile'**                   | Promedio de meses que la información crediticia del cliente ha estado en archivo.                    | Ayuda a entender la consistencia y estabilidad en el historial crediticio a lo largo del tiempo.                                                                    |
| **'NumSatisfactoryTrades'**            | Número de operaciones comerciales que el cliente ha manejado de manera satisfactoria.               | Indica la capacidad del cliente para manejar transacciones de manera exitosa.                                                                                        |
| **'NumTrades60Ever2DerogPubRec'**      | Número de operaciones comerciales en los últimos 60 meses con al menos dos registros derogatorios o públicos. | Indica la presencia de eventos negativos recientes.                                                                                                                |
| **'NumTrades90Ever2DerogPubRec'**      | Número de operaciones comerciales en los últimos 90 meses con al menos dos registros derogatorios o públicos. | Similar al anterior, pero en un periodo de tiempo más amplio.                                                                                                      |
| **'PercentTradesNeverDelq'**           | Porcentaje de transacciones comerciales en las que el cliente nunca ha incurrido en demoras.         | Representa la buena conducta de pago del cliente.                                                                                                                   |
| **'MSinceMostRecentDelq'**             | Meses desde la última demora en el pago.                                                           | Muestra el tiempo desde la última vez que el cliente no cumplió con los pagos.                                                                                     |
| **'MaxDelq2PublicRecLast12M'**        | Máxima demora atrasada a registros públicos en los últimos 12 meses.                                | Indica la gravedad de las demoras en los últimos 12 meses.                                                                                                          |
| **'MaxDelqEver'**                      | Máxima demora atrasada jamás registrada.                                                          | Refleja la demora más severa que el cliente ha experimentado.                                                                                                       |
| **'NumTotalTrades'**                   | Número total de operaciones comerciales en el historial crediticio.                                 | Representa la cantidad global de transacciones comerciales en la historia del cliente.                                                                             |
| **'NumTradesOpeninLast12M'**           | Número de operaciones comerciales abiertas en los últimos 12 meses.                                  | Indica cuántas cuentas comerciales ha abierto el cliente recientemente.                                                                                             |
| **'PercentInstallTrades'**            | Porcentaje de transacciones comerciales relacionadas con instalaciones crediticias.                 | Representa la proporción de transacciones asociadas a préstamos de instalación.                                                                                     |
| **'MSinceMostRecentInqexcl7days'**    | Meses desde la última consulta crediticia (excluyendo las realizadas en los últimos 7 días).        | Muestra el tiempo desde la última vez que se consultó la información crediticia.                                                                                    |
| **'NumInqLast6M'**                    | Número de consultas a la información crediticia en los últimos 6 meses.                              | Indica cuántas veces se ha revisado el historial crediticio en un periodo reciente.                                                                                |
| **'NumInqLast6Mexcl7days'**           | Número de consultas a la información crediticia en los últimos 6 meses (excluyendo las realizadas en los últimos 7 días). | Similar al anterior, pero excluyendo consultas muy recientes.                                                                                                       |
| **'NetFractionRevolvingBurden'**      | Fracción neta de la carga de crédito rotativo.                                                     | Representa la proporción de la deuda en tarjetas de crédito respecto al límite total.                                                                              |
| **'NetFractionInstallBurden'**        | Fracción neta de la carga de préstamos de instalación.                                             | Indica la proporción de la deuda en préstamos de instalación respecto al límite total.                                                                              |
| **'NumRevolvingTradesWBalance'**      | Número de operaciones comerciales rotativas con saldo pendiente.                                    | Indica cuántas cuentas de tarjetas de crédito tienen saldos pendientes.                                                                                             |
| **'NumInstallTradesWBalance'**        | Número de operaciones comerciales de instalación con saldo pendiente.                               | Similar al anterior, pero para cuentas de préstamos de instalación.                                                                                                 |
| **'NumBank2NatlTradesWHighUtilization'** | Número de operaciones comerciales de bancos frente a operaciones nacionales con alta utilización.  | Indica la presencia de cuentas de bancos con alto uso en comparación con cuentas nacionales.                                                                        |
| **'PercentTradesWBalance'**            | Porcentaje de transacciones comerciales con saldo pendiente.                                        | Representa el porcentaje de transacciones comerciales en las que el cliente mantiene un saldo pendiente.                                                            |

</div>

## 2. **Importación de paquetes, funciones y DataSet**<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1. Paquetes y Funciones<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se importan los paquetes y funciones utilizados en todo el proceso de análisis y modelado desde un script, siendo esta opción la adecuada para una presentación lo más limpia posible.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Las funciones utilizadas en el proceso de análisis pueden visualizarse en [este notebook](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/funciones_custom.ipynb).

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2. DataSet

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;El DataSet de trabajo es propiedad de [FICO](https://www.fico.com/en).<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Los datos se han obtenido en [huggingface](https://huggingface.co/datasets/mstz/heloc), gracias a [Mattia](https://huggingface.co/mstz).

## 3. **Análisis Exploratorio de Datos**<br>

En esta sección se exploran los datos para una mayor comprensión del problema a tratar. A su vez se estudian posibles inconsistencias que puedan tener los datos y su posterior tratamiento como veremos en la Sección 4.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1. Inspección de valores especiales<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Hemos podido observar valores negativos en algunas variables. Según la fuente de datos estos valores se dividen en 3 categorías:<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**Tipo 1**</u>: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cuando el informe de la agencia de crédito de un solicitante de préstamo no fue investigado o no se encontró, todas las características obtenidas del informe de la agencia de crédito reciben <u>**un valor especial de -9.**</u>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**Tipo 2**</u>: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Si la información recibida no es utilizable se anota <u>**un valor especial de -8.**</u>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**Tipo 3**</u>: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cuando no existe información del tipo solicitado se asigna <u>**un valor especial de -7.**</u>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Estos valores especiales parecen estar relacionados con situaciones en las que no se obtuvo información del informe de la agencia de crédito. Esta situación puede deberse a la confusión entre un solicitante VIP (cuyo informe podría no ser investigado) y la falta de un informe de la agencia de crédito (lo cual es un rasgo negativo para la extensión de crédito).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se tienen 588 filas sin información útil(todo valores especiales de Tipo 1), con un balance en la variable objetivo aproximadamente igual, por lo que se excluirán de nuestro análisis.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;El resto de valores especiales están localizados en diferentes columnas, siendo las cuatro con mayor cantidad de estos las siguientes:

<div align="center">

![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/col_val_esp.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En la Sección 4 se tratarán estos valores especiales, probando diferentes métodos de procesado y escogiendo el que mejores resultados proporcione.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2. Duplicados<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observan 587 valores duplicados en nuestros datos.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A partir de este punto se decide analizar las siguientes subsecciones sin tener en cuenta los valores especiales y los duplicados. Estos se tratarán el la Sección 4.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3. Análisis de Correlaciones<br>

<div align="center">

![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/corr.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observan ciertas columnas con una correlación muy alta entre ellas, siendo:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- La columna <u>'NumInqLast6Mexcl7days'</u> es la misma que la columna <u>'NumInqLast6M'</u> exceptuando esta los últimos 7 días, por tanto se tomará la columna con mayor información (<u>**'NumInqLast6M'**</u>).
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- De igual forma sucede con las columnas <u>'NumTrades60Ever2DerogPubRec'</u> y <u>'NumTrades90Ever2DerogPubRec'</u>, cuya diferencia radica en el intervalo temporal, siendo el del primero de 60 meses y el segundo de 90. Se tendrá en cuenta la columna con mayor rango temporal (<u>**'NumTrades90Ever2DerogPubRec'**</u>).

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4. Análisis de Outliers<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En este apartado se observa como hay columnas con datos muy dispersos, por tanto, en la Sección 4, eliminaremos los outliers con el método de Tukey, transformando algunas de las columnas a escala logarítmica dada su gran dispersión, tratanto de conseguir una distribución lo más normal posible y descartando los datos más extremos.


### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5. Análisis de Dispersión<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;El análisis de dispersión no muestra ninguna nube de puntos bien definida, concluyendo que no existen grupos diferenciados.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.6. Balance de clases

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;El DataSet no presenta desbalanceo de clases.

## 4. **Procesamiento de Datos**<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1. Columnas<br>
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1.1. Transformación columna objetivo<br>
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1.2. Cambio de nombre en columnas<br>
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1.3. Eliminación columnas altamente correlacionadas<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2. Eliminación de Duplicados<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.3. Tratamiento de valores especiales<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Para esta sección se han probado diferentes formas de tratar con los valores especiales:
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Se han tomando solo los datos positivos para realizar un estudio de la importancia de cada variable a traves de varios modelos, quedandonos con el Gradient Boosting Classifier. Los resultados arrojan que la columna con mayor cantidad de valores especiales es relevantes y por tanto no podríamos eliminarla. Para las demas columnas con mayor número valores especiales se prueba a descartarlas dada su baja importancia. Véase el proceso en el [**Anexo Sección 1**](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/anexo.ipynb).
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Hemos probado creando columnas booleanas para cada columna y cada valor especial, realizando un análisis de correlacion de estas columnas con la variable objetivo, sin encontrarse una relación fuerte. Por tanto se ha descartado esta opción como viable a la hora de tratar los valores especiales. Para visualizar los resultados obtenidos vease el [**Anexo Sección 2**](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/anexo.ipynb).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Se ha probado a crear una columna booleana a partir de la columna con más valores especiales, categorizando los valores positivos como True y los negativos como False. El resultado no ha sido favorable descartando esta opción como viable. Véase en el [**Anexo Sección 3**](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/anexo.ipynb).
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Se ha comprobado el resultado al imputar los valores especiales, arrojando, por el momento, los mejores resultados.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Se prueba, creando una columna booleana que asigne valor 1 a los datos que han sido imputados y 0 a los reales para las tres columnas con más valores especiales. No se observa ninguna mejora. Véase [**Anexo Sección 4**](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/anexo.ipynb).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<u>**Se ha decidido**</u>, tras múltiples pruebas, <u>**asignar NaN's a los valores especiales**</u>, imputándolos con KNNImputer a la hora de buscar el mejor modelo.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.4. Tratamiento de Outliers

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En esta sección los outliers se han tratado mediante el método de Tukey. Se han transformado algunas columnas a escala logarítmica devido a la gran dispersión de sus datos, tratando de normalizar la distribución y eliminar los valores más extremos.

## 5. **Entrenamiento de Modelo**<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.1. Selección de Modelo<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En esta subsección se escogen los siguientes modelos de clasificación con los que probar nuestros datos:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Logistic Regression

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Gaussian NB

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- KNeighbors Classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nearest Centroid

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Random Forest Classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Suport Vector Machine

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Ada Boost Classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Gradient Boosting Classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- XGB Classifier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Hist Gradient Boosting Classifier

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2. Cross Validation<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Para evaluar el rendimiento de nuestro modelo, emplearemos técnicas de validación cruzada. Esto implica dividir nuestro conjunto de datos en múltiples subconjuntos, entrenar el modelo en diferentes combinaciones de estos subconjuntos y evaluar su rendimiento. La validación cruzada ayuda a garantizar que nuestro modelo sea robusto y no se ajuste demasiado a los datos de entrenamiento.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Para elegir el mejor modelo de manera consistente se decide utilizar solo el conjunto de datos con valores positivos(sin tener en cuenta los valores especiales).

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2.1. Hold-Out<br>

<div align="center">

![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/hold_out.png)

</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2.2. k-Fold<br>

<div align="center">

![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/kfold.png)

</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2.3. Stratified k-Fold<br>

<div align="center">

![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/stra_kfold.png)

</div>

------------

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tras los <u>**resultados obtenidos**</u> se escogen los modelos que consideramos mejor se adaptan, <u>**Gradient Boosting Classifier**</u>, <u>**Random Forest Classifier**</u> y <u>**LogisticRegression**</u>.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.3. Mejor Modelo<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En esta subsección vamos a utilizar el conjunto de datos total, imputando los valores especiales con el KNNImputer. Para cada uno de los modelos se comprobará cual tiene un mejor rendimiento buscando el mejor número de vecinos, escogiendo el que arroje una mejor precisión.

<div align="center">

| Model                      | Accuracy | Precision | Recall |
|----------------------------|----------|-----------|--------|
| GradientBoostingClassifier(k=8) | 0.7287   | 0.7322    | 0.7417 |

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Los <u>**mejores resultados**</u> se obtienen para el modelo de <u>**Gradient Boosting Classifier**</u>, imputando los datos con <u>**8 vecinos**</u>. Se procede a continuación al tuning del modelo para intentar mejorar su rendimiento.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.4. Gradient Boosting Classifier<br>
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.4.1. Tuning<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tras varias pruebas con diferentes parámetros se tiene que <u>**los mejores resultados encontrados para el Gradient Boosting Classifier**</u> son los siguientes:

<div align="center">

| Model                      | Accuracy | Precision | Recall |
|----------------------------|----------|-----------|--------|
| GradientBoostingClassifier | 0.7384   | 0.7360    | 0.7630 |

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se ha estudiado el feature importance para este modelo y comprobando si se puede mejorar el rendimiento y disminuir la complejidad eliminando alguna variable. Para ello hemos evaluado el modelo modificando la cantidad de variables.

<div align="center">
  
![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/columnas.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa como los mejores resultados se obtienen teniendo en cuenta todas las variables en el modelo. Sin embargo, podría ser una opción razonable disminuir la complejidad de este ya que, teniendo en cuenta las 6 primeras columnas, se reduciría el accuracy entorno a un 1% y la cantidad de variables en más de un tercio, siendo esto una mejora considerable en la interpretabilidad de nuestro modelo.

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.4.2. Resultados<br>

<div align="center">
  
![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/matrix_modelo.png)

</div>

<div align="center">

| Accuracy | Precision | Recall |
|----------|-----------|--------|
| 0.7384   | 0.7360    | 0.7630 |

</div>


### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.5. Red Neuronal<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En esta sección se ha decidido probar con una fully connected neural network (FCNN), inspirada en el proyecto de [ali-ghorbani-k](https://github.com/ali-ghorbani-k/Credit-Risk-Management) en GitHub.

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.5.1. Resultados Red Neuronal

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tras los resultados del entrenamiento se observa como <u>**la perdida comienza a aumentar a partir de las 4 epochs**</u>, por lo que es en este punto cuando se decide parar ya que si no caeríamos en sobreentrenamiento.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Los mejores resultados se han obtenido para 4 epochs y 14 vecinos, siendo estos los siguientes:

<div align="center">

| Accuracy | Precision | Recall |
|----------|-----------|--------|
| 0.7425   | 0.7482    | 0.7497 |

</div>

<div align="center">
  
![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/red_graf.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Veamos como varían las metricas dependiendo del threshold que utilicemos:

<div align="center">
  
![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/thresh.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En un <u>**problema**</u> como el nuestro, donde se trata de <u>**decidir si entregar una linea de credito a un cliente dependiendo de su informe crediticio**</u>, tenemos que tener en cuenta nuestros intereses a la hora de hacer predicciones. Esto es:
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Si nuestra <u>**posición**</u>  es <u>**conservadora**</u> , buscaremos minimizar el riesgo de impago de un crédito y por tanto escogeremos con más cuidado a nuestros clientes, pudiendo perder a algunos que sean buenos pagadores. En este caso tenderemos a buscar un mayor <u>**Recall**</u> .
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Por el contrario, si nuestro <u>**enfoque**</u>  es más <u>**agresivo**</u> , buscaremos maximizar nuestras ganancias a expensas de incrementar el riesgo de impago, captando a más clientes aunque estos puedan incurrir en impagos y excluyendo a los que se sabe con mayor certeza que presentan problemas crediticios. En este caso buscaremos una mayor <u>**Precision**</u> 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;La <u>**matriz de confusión**</u> arroja la siguiente información para un <u>**threshold de 0.5**</u>:
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**Nuestras métricas están balanceadas**</u>, no presentando grandes diferencias entre sus valores, lo que nos dice que el modelo no sobrevalora en exceso una métrica sobre otras.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- De las <u>**predicciones**</u>  se obtiene que:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**685**</u> clientes han sido categorizados como <u>**verdaderos negativos**</u>  y <u>**247**</u>  como <u>**falsos negativos**</u> .
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <u>**734**</u> clientes han sido categorizados como <u>**verdaderos positivos**</u>  y <u>**245**</u>  como <u>**falsos positivos**</u> .

<div align="center">
  
![](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/images/matrix_red.png)

</div>

## 6. **Resultados Finales**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tras la evaluación del modelo <u>**Gradient Boosting Classifier**</u> y de la <u>**Red Neuronal**</u> podemos concluir que el uso de esta última consigue unos resultados un 1% mejores que los que consigue el modelo. Además, el tiempo de entrenamiento de la red es mucho menor al tiempo que ha tardado en ejecutarse el tunning de el modelo de Machine Learning, por lo que es otro punto a favor que nos hace concluír que la <u>**Red Neuronal se adapta mejor a la resolución de este problema**</u>.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En cuanto a las métricas obtenidas, vemos que ambos modelos consiguen un <u>**rendimiento**</u> entre el <u>**73-76% de accuracy**</u> lo que mejora los resultados durante la <u>**validación cruzada**</u> donde veíamos valores más cercanos al <u>**70%**</u>. En un problema balanceado de este tipo, estas métricas tan superiores al 50% indican que <u>**nuestros modelos han sido capaces de aprender**</u> patrones a partir de los datos, lo que era uno de los objetivos principales de este proyecto.

------------
    
- ## [**Anexo**](https://github.com/UrkoRegueiro/HACK-A-BOSS-PROJECTS/blob/main/HELOC_Project/anexo.ipynb)











