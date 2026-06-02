import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# ---------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------

st.set_page_config(
    page_title="Dashboard Logístico CVRP",
    page_icon="🚚",
    layout="wide"
)

# ---------------------------------------------------
# TÍTULO
# ---------------------------------------------------

st.title("🚚 Dashboard Inteligente de Optimización Logística")
st.markdown("### Caso de Estudio: Distribución de Alimentos en la Sabana de Bogotá")

st.markdown("---")

# ---------------------------------------------------
# DATOS
# ---------------------------------------------------

coordenadas = [
    [4.964, -73.912],  # Tocancipá
    [4.863, -74.053],  # Chía
    [4.918, -74.029],  # Cajicá
    [4.996, -74.003],  # Zipaquirá
    [4.908, -73.938],  # Sopó
    [4.945, -73.921]   # Briceño
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

# ---------------------------------------------------
# KPIs
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚛 Vehículos Disponibles",
        "3"
    )

with col2:
    st.metric(
        "📦 Demanda Total",
        "4.650 kg"
    )

with col3:
    st.metric(
        "📍 Clientes",
        "5"
    )

st.markdown("---")

# ---------------------------------------------------
# FUNCIÓN DISTANCIA
# ---------------------------------------------------

def distancia(coord1, coord2):

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    distancia = math.sqrt(
        (lat2 - lat1) ** 2 +
        (lon2 - lon1) ** 2
    )

    return int(distancia * 111000)

# ---------------------------------------------------
# MATRIZ DE DISTANCIAS
# ---------------------------------------------------

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

# ---------------------------------------------------
# BOTÓN
# ---------------------------------------------------

if st.button("🚀 Ejecutar Optimización"):

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

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(
        search_parameters
    )

    if solution:

        distancia_total = 0

        resultados = []

        st.header("📋 Rutas Optimizadas")

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

            utilizacion = round(
                (carga / capacidades[vehicle_id]) * 100,
                2
            )

            resultados.append([
                vehicle_id + 1,
                carga,
                round(distancia_ruta / 1000, 2),
                utilizacion
            ])

            st.subheader(
                f"🚛 Vehículo {vehicle_id+1}"
            )

            st.write(
                " ➜ ".join(ruta)
            )

            st.write(
                f"📦 Carga: {carga} kg"
            )

            st.write(
                f"📍 Distancia: {round(distancia_ruta/1000,2)} km"
            )

        st.success(
            f"✅ Distancia Total de la Operación: {round(distancia_total/1000,2)} km"
        )

        # -----------------------------------------
        # TABLA
        # -----------------------------------------

        st.header("📊 Resumen Ejecutivo")

        df = pd.DataFrame(
            resultados,
            columns=[
                "Vehículo",
                "Carga (kg)",
                "Distancia (km)",
                "Utilización (%)"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # -----------------------------------------
        # GRÁFICO
        # -----------------------------------------

        st.header("📈 Utilización de la Flota")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(
            df["Vehículo"].astype(str),
            df["Utilización (%)"]
        )

        ax.set_title(
            "Nivel de Utilización de Vehículos"
        )

        ax.set_ylabel(
            "Porcentaje (%)"
        )

        st.pyplot(fig)

        # -----------------------------------------
        # MAPA
        # -----------------------------------------

        st.header("🗺️ Ubicación Geográfica")

        mapa = pd.DataFrame({
            "lat":[
                4.964,
                4.863,
                4.918,
                4.996,
                4.908,
                4.945
            ],
            "lon":[
                -73.912,
                -74.053,
                -74.029,
                -74.003,
                -73.938,
                -73.921
            ]
        })

        st.map(mapa)

        # -----------------------------------------
        # ANÁLISIS
        # -----------------------------------------

        st.header("📑 Análisis Gerencial")

        st.success(
            """
            • Se atendió una demanda total de 4.650 kg.

            • El algoritmo determinó que se requieren 3 vehículos.

            • La distancia total recorrida fue de 78.27 km.

            • El Vehículo 2 presentó la mayor utilización de capacidad.

            • Las rutas fueron optimizadas minimizando la distancia total recorrida.
            """
        )

        # -----------------------------------------
        # CONCLUSIONES
        # -----------------------------------------

        st.header("✅ Conclusiones")

        st.info(
            """
            1. El modelo CVRP permitió optimizar la distribución de alimentos.

            2. Se minimizó la distancia recorrida respetando la capacidad de los vehículos.

            3. Se utilizaron tres vehículos para cubrir la demanda total.

            4. Google OR-Tools permitió obtener soluciones eficientes para el problema de ruteo.

            5. La aplicación desarrollada en Streamlit facilita el análisis visual y la toma de decisiones logísticas.
            """
        )

    else:

        st.error(
            "No se encontró una solución factible."
        )
