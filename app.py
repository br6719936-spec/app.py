import streamlit as st
import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

st.set_page_config(page_title="CVRP Sabana de Bogotá")

st.title("🚚 Optimización de Rutas - CVRP")
st.write("Caso 1: Distribución de alimentos en la Sabana de Bogotá")

coordenadas = [
    [4.964, -73.912],
    [4.863, -74.053],
    [4.918, -74.029],
    [4.996, -74.003],
    [4.908, -73.938],
    [4.945, -73.921]
]

nombres = [
    "CEDI Tocancipá",
    "Chía",
    "Cajicá",
    "Zipaquirá",
    "Sopó",
    "Briceño"
]

demandas = [0, 1100, 750, 1400, 900, 500]
capacidades = [2200, 2200, 2200]

def distancia(c1, c2):
    return int(
        math.sqrt(
            (c2[0]-c1[0])**2 +
            (c2[1]-c1[1])**2
        ) * 111000
    )

matriz = []

for i in range(len(coordenadas)):

    fila = []

    for j in range(len(coordenadas)):

        fila.append(
            distancia(
                coordenadas[i],
                coordenadas[j]
            )
        )

    matriz.append(fila)

if st.button("Optimizar Rutas"):

    manager = pywrapcp.RoutingIndexManager(
        len(matriz),
        len(capacidades),
        0
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
        "Capacity"
    )

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(
        search_parameters
    )

    if solution:

        distancia_total = 0

        for vehicle_id in range(len(capacidades)):

            index = routing.Start(vehicle_id)

            ruta = []
            carga = 0
            distancia_ruta = 0

            while not routing.IsEnd(index):

                node = manager.IndexToNode(index)

                ruta.append(nombres[node])

                carga += demandas[node]

                previous_index = index

                index = solution.Value(
                    routing.NextVar(index)
                )

                distancia_ruta += routing.GetArcCostForVehicle(
                    previous_index,
                    index,
                    vehicle_id
                )

            ruta.append("CEDI Tocancipá")

            distancia_total += distancia_ruta

            st.subheader(f"Vehículo {vehicle_id+1}")

            st.write(
                " ➜ ".join(ruta)
            )

            st.write(
                f"Carga: {carga} kg"
            )

            st.write(
                f"Distancia: {round(distancia_ruta/1000,2)} km"
            )

        st.success(
            f"Distancia total: {round(distancia_total/1000,2)} km"
        )
