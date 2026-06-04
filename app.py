# ============================================
# DASHBOARD LOGÍSTICO CVRP - GOOGLE COLAB
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import math
import folium

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# ===============================
# DATOS
# ===============================

datos = pd.DataFrame({
    "Nombre": [
        "CEDI Tocancipá",
        "Chía",
        "Cajicá",
        "Zipaquirá",
        "Sopó",
        "Briceño"
    ],
    "Latitud": [4.964,4.863,4.918,4.996,4.908,4.945],
    "Longitud": [-73.912,-74.053,-74.029,-74.003,-73.938,-73.921],
    "Demanda (kg)": [0,1100,750,1400,900,500]
})

DEPOSITO = 0
NUM_VEHICULOS = 3
CAPACIDAD_VEHICULO = 2200
TIEMPO_BUSQUEDA = 5

# ===============================
# FUNCIONES
# ===============================

def distancia_haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371000

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat/2)**2 +
         math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)

    c = 2 * math.asin(math.sqrt(a))

    return int(round(R * c))

def construir_matriz_distancias(df):

    coordenadas = df[["Latitud","Longitud"]].values.tolist()

    matriz = []

    for i in range(len(coordenadas)):
        fila = []

        for j in range(len(coordenadas)):

            if i == j:
                fila.append(0)
            else:
                fila.append(
                    distancia_haversine(
                        coordenadas[i],
                        coordenadas[j]
                    )
                )

        matriz.append(fila)

    return matriz

def resolver_cvrp(matriz, demandas, capacidades, deposito, segundos):

    manager = pywrapcp.RoutingIndexManager(
        len(matriz),
        len(capacidades),
        deposito
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):

        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return matriz[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    def demand_callback(from_index):

        from_node = manager.IndexToNode(from_index)

        return demandas[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        capacidades,
        True,
        "Capacidad"
    )

    parametros = pywrapcp.DefaultRoutingSearchParameters()

    parametros.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    parametros.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    parametros.time_limit.seconds = segundos

    solution = routing.SolveWithParameters(parametros)

    return manager, routing, solution

# ===============================
# EJECUCIÓN
# ===============================

nombres = datos["Nombre"].tolist()
demandas = datos["Demanda (kg)"].tolist()
capacidades = [CAPACIDAD_VEHICULO] * NUM_VEHICULOS

print("DATOS DEL PROBLEMA")
display(datos)

matriz = construir_matriz_distancias(datos)

print("\\nMATRIZ DE DISTANCIAS (km)")
display(
    pd.DataFrame(
        matriz,
        index=nombres,
        columns=nombres
    ) / 1000
)

manager, routing, solution = resolver_cvrp(
    matriz,
    demandas,
    capacidades,
    DEPOSITO,
    TIEMPO_BUSQUEDA
)

if not solution:
    print("No se encontró solución.")
else:

    resumen = []
    distancia_total = 0

    mapa = folium.Map(
        location=[
            datos["Latitud"].mean(),
            datos["Longitud"].mean()
        ],
        zoom_start=10
    )

    for _, fila in datos.iterrows():
        folium.Marker(
            [fila["Latitud"], fila["Longitud"]],
            popup=fila["Nombre"]
        ).add_to(mapa)

    for vehicle_id in range(NUM_VEHICULOS):

        index = routing.Start(vehicle_id)

        ruta_nombres = []
        ruta_indices = []

        carga = 0
        distancia = 0

        while not routing.IsEnd(index):

            nodo = manager.IndexToNode(index)

            ruta_indices.append(nodo)
            ruta_nombres.append(nombres[nodo])

            carga += demandas[nodo]

            previo = index

            index = solution.Value(
                routing.NextVar(index)
            )

            distancia += routing.GetArcCostForVehicle(
                previo,
                index,
                vehicle_id
            )

        ruta_indices.append(DEPOSITO)
        ruta_nombres.append(nombres[DEPOSITO])

        if len(ruta_indices) <= 2:
            continue

        distancia_total += distancia

        print("\\n" + "="*50)
        print(f"VEHÍCULO {vehicle_id+1}")
        print("="*50)
        print("Ruta:", " ➜ ".join(ruta_nombres))
        print("Carga:", carga, "kg")
        print("Distancia:", round(distancia/1000,2), "km")

        resumen.append({
            "Vehículo": vehicle_id + 1,
            "Carga (kg)": carga,
            "Distancia (km)": round(distancia/1000,2),
            "Utilización (%)":
                round(carga/CAPACIDAD_VEHICULO*100,2)
        })

        coordenadas = []

        for idx in ruta_indices:
            coordenadas.append([
                datos.loc[idx,"Latitud"],
                datos.loc[idx,"Longitud"]
            ])

        folium.PolyLine(
            coordenadas,
            weight=4
        ).add_to(mapa)

    print("\\nDISTANCIA TOTAL:", round(distancia_total/1000,2), "km")

    resumen_df = pd.DataFrame(resumen)

    display(resumen_df)

    plt.figure(figsize=(8,4))

    plt.bar(
        resumen_df["Vehículo"].astype(str),
        resumen_df["Utilización (%)"]
    )

    plt.title("Utilización de Vehículos")
    plt.xlabel("Vehículo")
    plt.ylabel("Utilización (%)")
    plt.ylim(0,100)

    plt.show()

    mapa
