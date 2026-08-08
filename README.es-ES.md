

# AutoDL
Un desafío de datos en Aprendizaje Profundo Automatizado (AutoDL), organizado por ChaLearn, Google y 4Paradigm.

[![Build Status](https://travis-ci.org/zhengying-liu/autodl.svg?branch=master)](https://travis-ci.org/zhengying-liu/autodl)

### Cómo contribuir
[Ver instrucciones](https://github.com/zhengying-liu/autodl/blob/master/CONTRIBUTING.md)

### Para preparar un paquete de competición y crear una copia de la competición AutoDL
Ejecute:
```
git clone https://github.com/zhengying-liu/autodl.git
cd autodl/codalab_competition_bundle/utilities/
./make_competition_bundle.sh
```
A continuación, verá que se crea un archivo zip en el directorio `utilities/`. Súballo a un servidor de CodaLab (como [este](http://autodl.lri.fr/)) en la etiqueta 'Create Competition' y listo.

### Consejos
Este paquete de competición debe ejecutarse en una instancia especializada de CodaLab como [esta](http://autodl.lri.fr/) y NO en el [servidor habitual](https://competitions.codalab.org/).

Las principales diferencias de la instancia especializada de CodaLab (en comparación con la habitual) son:
1. **Vías paralelas** se implementan utilizando una fase padre y varias fases hijas. 
Mientras se realiza una presentación en la fase padre, se creará una copia de esta presentación para *cada* fase hija. 
Por ejemplo, supongamos que tenemos una fase padre llamada 'All datasets', que tiene 5 fases hijas 'Dataset 1', 'Dataset 2', ..., 'Dataset 5'. 
Entonces, cuando se realiza una presentación en 'All datasets', se crearán 5 copias de esta presentación en las 5 fases hijas, lo que genera 5+1=6 *trabajos* (también conocidos como *ejecuciones*). 
Cada trabajo puede tener un paso de ingestión y/o un paso de puntuación. 
Los trabajos hijos se encargan por varios nodos de trabajo de forma paralela y asíncrona. 
Cuando los 5 trabajos hijos finalicen, el trabajo padre recibirá sus resultados (salida) y luego se ejecutará.
2. En lugar de tener varias tareas (definidas por varios conjuntos de datos) en un solo trabajo, ahora solo se encuentra **una tarea** (por lo tanto, un conjunto de datos) en cada trabajo. 
Por lo tanto, para preparar los archivos zip de los conjuntos de datos, solo necesita comprimir el contenido (no el directorio) de un directorio:
```
./
├── munster.data/
├── public.info (opcional)
```
para *Datos de entrada* y el contenido de un directorio:
```
./
├── munster.solution
```
para *Datos de referencia*. Por lo tanto, se deben crear y subir a CodaLab 10 archivos zip (5 archivos zip de datos + 5 archivos zip de soluciones) para las 5 fases hijas. Tenga en cuenta que esto es diferente del servidor habitual, donde los datos de varias tareas se comprimen en un solo archivo zip (y otro archivo zip para todas las soluciones).

3. El programa de ingestión y el programa de puntuación ahora pueden ejecutarse **en paralelo**, siempre que se active la función *Ingestión solo durante la puntuación* en la página del Editor de la competición (o en el archivo yaml de la competición). 
Cuando esta función está activada, la puntuación se iniciará al mismo tiempo que la ingestión. 
Si la ingestión excede el límite de tiempo asignado, la puntuación finalizará la ingestión utilizando su PID.

4. La retroalimentación en **tiempo real** se implementa mediante la página `detailed_results.html` generada por el programa de puntuación. 
Esta página se puede ver haciendo clic en el botón 'Learning Curve' en la pestaña 'My Submissions' o en el botón 'Detailed Results' en los resultados detallados de cada presentación. Esta función es útil para que los participantes vean el rendimiento de su algoritmo incluso si el trabajo aún no ha finalizado.

### Enlaces útiles:
- Versión actual de la competición en [CodaLab](https://autodl.lri.fr/)
- Información sobre [Protocol Buffers](https://developers.google.com/protocol-buffers/)
- Definición de [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
