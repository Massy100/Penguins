# Se requiere la instalacion de los siguientes paquetes
#install.packages(c('tidyverse','caret','neuralnet', 'palmerpenguins'))

#Cargar paquetes 
library(tidyverse)
library(caret)
library(neuralnet)
library(palmerpenguins)

#cargar conjunto de datos
datos = penguins

#Eliminar filas con valores NA (penguins tiene algunos datos faltantes)
datos = na.omit(datos)

#separacion de los datos en conjunto de entrenamiento y pruebas
set.seed(123) #para reproducibilidad
muestra = createDataPartition(datos$species, p=0.8, list = F)
train = datos[muestra,]
test = datos[-muestra,]

#analisis exploratorio
head(train,5)
tail(train,5)
train[17:25,]

#Variables disponibles en penguins: 
#bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, species

bill_length = train$bill_length_mm
hist(bill_length, main = "Histograma de Longitud del Pico", xlab = "Longitud (mm)")
hist(train$flipper_length_mm, main = "Histograma de Longitud de Aleta", xlab = "Longitud (mm)")

#Entrenamiento de red neuronal
#Nota: neuralnet requiere variables numéricas, por lo que convertimos species a factor y luego a dummy
red.neuronal = neuralnet(species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, 
                         data = train, 
                         hidden = c(2,3),
                         linear.output = FALSE)

red.neuronal$act.fct
plot(red.neuronal)

#Aplicar la red neuronal al conjunto de pruebas
prediccion = predict(red.neuronal, test, type='class')

#Decodificar maximo = Especie
specie.decod = apply(prediccion, 1, which.max)
specie.pred = data.frame(specie.decod)

#Las especies en penguins son: "Adelie", "Chinstrap", "Gentoo"
specie.pred = mutate(specie.pred, especie = recode(specie.pred$specie.decod, 
                                                   "1" = "Adelie", 
                                                   "2" = "Chinstrap", 
                                                   "3" = "Gentoo"))

test$Species.pred = specie.pred$especie

#Ver matriz de confusión para evaluar el modelo
confusionMatrix(as.factor(test$Species.pred), as.factor(test$species))