import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="Sistema Inteligente de Optimización Logística",
                   page_icon="🚚", layout="wide")

st.sidebar.title("🚚 Navegación")
seccion = st.sidebar.radio(
    "Seleccione una sección",
    ["Inicio", "Datos del Caso", "Optimización"]
)

coordenadas = [
    [4.964, -73.912],
    [4.863, -74.053],
    [4.918, -74.029],
    [4.996, -74.003],
    [4.908, -73.938],
    [4.945, -73.921]
]

nombres = ["CEDI Tocancipá","Chía","Cajicá","Zipaquirá","Sopó","Briceño"]
demandas = [0,1100,750,1400,900,500]
capacidades = [2200,2200,2200]

if seccion == "Inicio":
    st.title("🚚 Sistema Inteligente de Optimización Logística")
    st.markdown("""
    ## Proyecto Final - CVRP
    
    Herramientas:
    - Python
    - Streamlit
    - Google OR-Tools
    - Folium
    
    Objetivo:
    Minimizar la distancia total recorrida respetando la capacidad de los vehículos.
    """)
    st.stop()

if seccion == "Datos del Caso":
    st.header("📊 Datos del Caso")
    datos = pd.DataFrame({
        "Ubicación": nombres,
        "Demanda (kg)": demandas
    })
    st.dataframe(datos, use_container_width=True)
    st.stop()

st.title("🚚 Dashboard Inteligente de Optimización Logística")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Vehículos", "3")
col2.metric("Demanda", "4.650 kg")
col3.metric("Clientes", "5")
col4.metric("Capacidad", "2.200 kg")

def distancia(c1, c2):
    return int(math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2) * 111000)

matriz = []
for i in range(len(coordenadas)):
    fila = []
    for j in range(len(coordenadas)):
        fila.append(distancia(coordenadas[i], coordenadas[j]))
    matriz.append(fila)

if st.button("🚀 Ejecutar Optimización"):

    manager = pywrapcp.RoutingIndexManager(len(matriz), len(capacidades), 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return matriz[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demandas[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, capacidades, True, "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:

        resultados = []
        distancia_total = 0

        for vehicle_id in range(len(capacidades)):
            index = routing.Start(vehicle_id)
            carga = 0
            distancia_ruta = 0
            ruta = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                ruta.append(nombres[node])
                carga += demandas[node]

                previous_index = index
                index = solution.Value(routing.NextVar(index))

                distancia_ruta += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            ruta.append("CEDI Tocancipá")

            utilizacion = round((carga / capacidades[vehicle_id]) * 100, 2)

            resultados.append([
                vehicle_id + 1,
                carga,
                round(distancia_ruta / 1000, 2),
                utilizacion
            ])

            distancia_total += distancia_ruta

            st.subheader(f"Vehículo {vehicle_id+1}")
            st.write(" ➜ ".join(ruta))

        st.success(
            f"Distancia Total: {round(distancia_total/1000,2)} km"
        )

        df = pd.DataFrame(
            resultados,
            columns=[
                "Vehículo",
                "Carga (kg)",
                "Distancia (km)",
                "Utilización (%)"
            ]
        )

        st.header("📊 Resumen Ejecutivo")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Descargar Resultados CSV",
            csv,
            "resultados_cvrp.csv",
            "text/csv"
        )

        st.header("📈 Utilización de la Flota")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df["Vehículo"].astype(str), df["Utilización (%)"])
        ax.set_ylabel("Utilización (%)")
        st.pyplot(fig)

        st.header("🗺️ Mapa Interactivo")

        m = folium.Map(location=[4.94, -73.97], zoom_start=11)

        for i in range(len(coordenadas)):
            folium.Marker(
                location=coordenadas[i],
                popup=nombres[i],
                tooltip=nombres[i]
            ).add_to(m)

        st_folium(m, width=900, height=500)

        st.header("💼 Impacto Empresarial")
        st.success("""
        ✅ Reducción de costos de transporte.
        ✅ Mejor utilización de la flota.
        ✅ Optimización de rutas.
        ✅ Apoyo a la toma de decisiones.
        """)

        st.header("✅ Conclusiones")
        st.info("""
        - Se utilizaron 3 vehículos.
        - Se atendieron 4.650 kg de demanda.
        - Distancia total aproximada: 78.27 km.
        - Se optimizaron las rutas mediante CVRP.
        """)

    else:

        st.error(
            "No se encontró una solución factible."
        )
