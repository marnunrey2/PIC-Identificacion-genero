# NOTAS DEL PAPER DE oBIFs

## INTRODUCCION: 

- Se utilizan imagenes de escritura fem y masc para entrenar un SVM
- Neurocientificos y psicologos coinciden en que existe una relación en la manera de escribir segun atributos demograficos incluyendo el genero (bibliografia -> documentos mencionados)
- Otros estudios acerca de concluir que la letra de las mujeres es mejor ->  ( Burr, 2002; Hartley, 1991 )
- Los researchers dicen que la diferencia entre la letra de hombres y mujeres esta en el control motos y las hormonas (bibliografia -> documentos mencionados)

- Usan: una tecnica basada en 'oriented Basic Image Fetures (oBIF)' utilizando SVM como clasificador
- Computan: oBIF extraction histogram  y oBIF columns histogram; y al juntarlas forman el feature vector


## RELATED WORK:

Documentos que estudian lo mismo:
- Goodenough (1945) -> punto de vista de los psicologos
- Hartley (1991) -> conventional features
- Hayes (1996) -> impacto de las hormonas
- Hamid & Loewenthal (1996) -> human analysts
-  Al-Maadeed & Hassaine (2014); Bandi & Srihari (2005); Liwicki et al.
(2011); Siddiqi et al. (2012); Sokic et al. (2012) -> computerized analysis
- Liwicki et al. (2011) -> 67% accuracy
- Bouadjenek, Nemmour and Chibani (2014) -> Histogram of Oriented Gradients (HOG) y Local Binary Paterns (LBP) applied with SVM -> 74% accuracy
- Tabla pag 4

- ICDAR 2013 -> Hassaine, Al-Maadeed, Aljaam & Jaoua 2013 (71% accuracy)
- ICDAR 2015 -> Djeddi et al. 2015 (76% accuracy) 
- ICFHR 2016 -> Djeddi et al. 2016 (68% accuracy)


## OBIF BASED GNEDER CLASSIFICATION

- Primero se binariza la imagen con 'global thresholding'; luego se extraen features de el oBIF histogram y del oBIF colunms histogram; por ultimo se entrena el SVM con estas features
- Foto pag 3

- FEATURE EXTRACTION
- Otras veces se han usado el HOG, LBP y sus variantes y Gabor filters pero en este estudio vamos a utilizar oBIFs

- oBIFs HISTOGRAM
- oBIFs representa un texture-based descriptor que extiende a BIFs
- Cada pixel se clasifica segun simetria local utilizando el filtro Derivative-of-Gaussian (en segundo orden) σ: flat, slope, dark rotational, light rotational, dark line on light, light line on dark o saddle like (ver foto pag 5)
- ε representa el likelihood para que el pixel sea flat
- no entiendo la feature q cogen del histograma

- oBIFs COLUNM HISTOGRAM
- ignoramos el flat type, por lo que obtenemos la formula anterior (que no he entendido) al cuadrado y ese es el valor de histograma columna
- σ = {1,2,4,8,16}
- ε = {0.1, 0.01, 0.001}

- Juntamos ambos features y nos sale el vector que representa la imagen -> (h, ch)

- CLASIFICACION
- pasamos esas features al SVM 


## SYSTEM EVALUATION

- Script-dependent evaluations -> usado en las competiciones mencionadas 
- Script-independent evaluations -> es mas dificil, pero oBIFs es efectivo en encrontrar las caracteristicas comunes independientemente del script. Para hacerlo simplemente entrenas al SVM con arabic samples y haces el tests con english samples y luego al reves (el accuracy es menor obviamente)
