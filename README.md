# Face-Detector-Unity

Programa de detección de rostros y gestos de la mano.

Salida: Valores flotantes de dirección y aceleración. Se envían por TCP aL juego en Unity.

## Requerimientos:
- ```pip install mediapipe```
- ```pip install opencv-python```

## Modo de uso:
- Los detectores de rostro y de los gestos de la mano funcionan al mismo tiempo, por lo que la mano a detectar no debe tapar el rostro para no estropear la detección.
- La detección de rostro puede ser realizada con el rostro de la persona a una distancia de 40 ó 50 cm de la cámara aproximadamente. 
Solo se requiere mover la cabeza en la dirección que se quiera que se mueva la pelota.
- Para la detección de los gestos de la mano se requiere una distancia similar al de la detección de rostros, pero sin tapar la cara a detectar y mostrando la mano de frente a la cámara y no de costado.
- Si no ocurre ninguna detección se devolverán valores nulos.
