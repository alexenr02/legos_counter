import math

# Clase  para implementar rastreo de multiples objetos
# Usa el algoritmo de rastreo de centroides usando distancias euclideanas
class EuclideanDistTracker:
    def __init__(self):
        # Almacena los centroides
        self.center_points = {}
        # Almacena el conteo de ID's: cada que un objeto es detectado, la cuenta aumenta en 1
        self.id_count = 1


    #funcion que acepta un arreglo de coordenadas de contornos de rectangulos
    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Calcular ccentroides de los objetos deteectaos usando coordenadas de contornos de caja
        for rect in objects_rect:
            x, y, w, h = rect   #coordenadas rectangulo
            cx = (x + x + w) // 2 #centro
            cy = (y + y + h) // 2 #centro

            # Identificar si el mismo objeto ya ha sido detectado
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1]) #calcular la distancia euclideana entre el par de centroides


                #Identificando al mismo objeto
                #if dist < 350: #main.py
                if dist < 850:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Un objeto nuevo ha sido detectado y se le asigna un nuevo ID
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        # Limpia los centroides del diccionario para borrar los ID's que ya no se usan
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        # Actualizar diccionario con ID's que se borraron
        self.center_points = new_center_points.copy()

        #retorna una lista que contiene las coordenadas de los rectangulos (x,y,w,h) y el 
        #object_id es el id que es asignado a ese rectangulo en particular
        return objects_bbs_ids



