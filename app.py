import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

try:
    import pydeck as pdk
except ImportError:
    pdk = None

# ===================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ===================================================
st.set_page_config(
    page_title="Dashboard Logístico CVRP", page_icon="🚚", layout="wide"
)

# Inicializar estados de sesión para evitar borrados accidentales al interactuar
if "optimizado" not in st.session_state:
    st.session_state.optimizado = False
if "resultado" not in st.session_state:
    st.session_state.resultado = None

# ===================================================
# 2. VALORES POR DEFECTO Y CONSTANTES
# ===================================================
DEPOSITO = 0

DATOS_INICIALES = pd.DataFrame(
    {
        "Nombre": [
            "CEDI Tocancipá",
            "Chía",
            "Cajicá",
            "Zipaquirá",
            "Sopó",
            "Briceño",
        ],
        "Latitud": [4.964, 4.863, 4.918, 4.996, 4.908, 4.945],
        "Longitud": [-73.912, -74.053, -74.029, -74.003, -73.938, -73.921],
        "Demanda (kg)": [0, 1100, 750, 1400, 900, 500],
    }
)


# ===================================================
# 3. FUNCIONES LÓGICAS Y DE OPTIMIZACIÓN
# ===================================================
def distancia_haversine(coord1, coord2):
    """Calcula la distancia real aproximada entre dos coordenadas (en metros)."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    radio_tierra = 6371000
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return int(round(radio_tierra * c))


@st.cache_data(show_spinner=False)
def construir_matriz_distancias(df):
    """Construye la matriz de distancias en metros para todos los puntos."""
    coordenadas = df[["Latitud", "Longitud"]].values.tolist()
    matriz = []

    for i in range(len(coordenadas)):
        fila = []
        for j in range(len(coordenadas)):
            if i == j:
                fila.append(0)
            else:
                fila.append(distancia_haversine(coordenadas[i], coordenadas[j]))
        matriz.append(fila)
    return matriz


def resolver_cvrp(matriz, demandas, capacidades, deposito, segundos, nombres):
    """Resuelve el problema CVRP usando Google OR-Tools."""
    if sum(demandas) > sum(capacidades):
        return None, "La demanda total supera la capacidad total disponible."

    manager = pywrapcp.RoutingIndexManager(
        len(matriz), len(capacidades), deposito
    )
    routing = pywrapcp.RoutingModel(manager)

    # Callbacks de tránsito y demanda
    def distance_callback(from_index, to_index):
        return matriz[manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)
        ]

    def demand_callback(from_index):
        return demandas[manager.IndexToNode(from_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, capacidades, True, "Capacidad"
    )

    # Parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(segundos)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, "No se encontró una solución factible."

    # Procesamiento de resultados
    rutas = []
    distancia_total = 0

    for vehicle_id in range(len(capacidades)):
        index = routing.Start(vehicle_id)
        ruta_indices = []
        carga = 0
        distancia_ruta = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            ruta_indices.append(node)
            carga += demandas[node]

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            distancia_ruta += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        ruta_indices.append(deposito)
        clientes_visitados = [i for i in ruta_indices if i != deposito]

        if not clientes_visitados:
            continue

        distancia_total += distancia_ruta
        rutas.append(
            {
                "Vehículo": vehicle_id + 1,
                "Ruta índices": ruta_indices,
                "Ruta": " ➜ ".join([nombres[i] for i in ruta_indices]),
                "Carga (kg)": carga,
                "Distancia (km)": round(distancia_ruta / 1000, 2),
                "Utilización (%)": round(
                    (carga / capacidades[vehicle_id]) * 100, 2
                ),
            }
        )

    return {
        "rutas": rutas,
        "distancia_total_km": round(distancia_total / 1000, 2),
    }, None


def crear_segmentos_ruta(df, rutas):
    """Genera los segmentos espaciales para la visualización en PyDeck."""
    segmentos = []
    for ruta in rutas:
        indices = ruta["Ruta índices"]
        for origen, destino in zip(indices[:-1], indices[1:]):
            segmentos.append(
                {
                    "Vehículo": f"Vehículo {ruta['Vehículo']}",
                    "Origen": df.loc[origen, "Nombre"],
                    "Destino": df.loc[destino, "Nombre"],
                    "path": [
                        [df.loc[origen, "Longitud"], df.loc[origen, "Latitud"]],
                        [
                            df.loc[destino, "Longitud"],
                            df.loc[destino, "Latitud"],
                        ],
                    ],
                }
            )
    return pd.DataFrame(segmentos)


# ===================================================
# 4. INTERFAZ DE USUARIO (STREAMLIT)
# ===================================================

# Encabezados principales
st.title("🚚 Dashboard Inteligente de Optimización Logística")
st.markdown("### Caso de Estudio: Distribución de Alimentos en la Sabana de Bogotá")
st.markdown("---")

# Panel Lateral de Configuración
st.sidebar.header("⚙️ Parámetros del Modelo")
num_vehiculos = st.sidebar.number_input(
    "Número de vehículos", min_value=1, max_value=10, value=3, step=1
)
capacidad_vehiculo = st.sidebar.number_input(
    "Capacidad por vehículo (kg)",
    min_value=100,
    max_value=10000,
    value=2200,
    step=100,
)
tiempo_busqueda = st.sidebar.slider(
    "Tiempo máximo de optimización (segundos)", min_value=1, max_value=30, value=5
)

capacidades = [int(capacidad_vehiculo)] * int(num_vehiculos)

# Editor de datos en el cuerpo principal
with st.expander("✏️ Editar datos de clientes y demandas"):
    puntos = st.data_editor(
        DATOS_INICIALES, use_container_width=True, num_rows="fixed"
    )

# Saneamiento de datos post-edición
puntos = puntos.reset_index(drop=True)
puntos.loc[DEPOSITO, "Demanda (kg)"] = 0
nombres = puntos["Nombre"].tolist()
demandas = puntos["Demanda (kg)"].astype(int).tolist()

# KPIs Informativos de la parte superior
demanda_total = int(sum(demandas))
capacidad_total = int(sum(capacidades))
clientes = len(puntos) - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("🚛 Vehículos disponibles", num_vehiculos)
col2.metric("📦 Demanda total", f"{demanda_total:,} kg")
col3.metric("🏋️ Capacidad total", f"{capacidad_total:,} kg")
col4.metric("📍 Clientes", clientes)
st.markdown("---")

# Validaciones de consistencia
if demanda_total > capacidad_total:
    st.error(
        "⚠️ La demanda total supera la capacidad disponible. "
        "Aumenta el número de vehículos o la capacidad por vehículo."
    )

# Matriz de distancias oculta por defecto
matriz = construir_matriz_distancias(puntos)
with st.expander("📏 Ver matriz de distancias aproximadas en km"):
    matriz_km = pd.DataFrame(matriz, columns=nombres, index=nombres) / 1000
    st.dataframe(matriz_km.round(2), use_container_width=True)

# Disparador de optimización
disabled_btn = demanda_total > capacidad_total
if st.button("🚀 Ejecutar Optimización", disabled=disabled_btn):
    with st.spinner("Optimizando rutas..."):
        resultado, error = resolver_cvrp(
            matriz=matriz,
            demandas=demandas,
            capacidades=capacidades,
            deposito=DEPOSITO,
            segundos=tiempo_busqueda,
            nombres=nombres,
        )

        if error:
            st.error(error)
            st.session_state.optimizado = False
        else:
            st.session_state.resultado = resultado
            st.session_state.optimizado = True

# ===================================================
# 5. RENDERIZADO DE RESULTADOS Y GRÁFICOS
# ===================================================
if st.session_state.optimizado and st.session_state.resultado:
    rutas = st.session_state.resultado["rutas"]
    distancia_total_km = st.session_state.resultado["distancia_total_km"]

    st.header("📋 Rutas Optimizadas")
    for r in rutas:
        st.subheader(f"🚛 Vehículo {r['Vehículo']}")
        st.write(f"**Ruta:** {r['Ruta']}")
        st.write(f"📦 **Carga:** {r['Carga (kg)']} kg")
        st.write(f"📍 **Distancia:** {r['Distancia (km)']} km")
        st.write(f"⚙️ **Utilización:** {r['Utilización (%)']} %")
        st.markdown("---")

    st.success(f"✅ Distancia total de la operación: {distancia_total_km} km")

    # Sección de Resumen Ejecutivo y Descargas
    st.header("📊 Resumen Ejecutivo")
    df_resumen = pd.DataFrame(rutas)
    df_tabla = df_resumen[
        ["Vehículo", "Carga (kg)", "Distancia (km)", "Utilización (%)", "Ruta"]
    ]
    st.dataframe(df_tabla, use_container_width=True)

    csv = df_tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resumen en CSV",
        data=csv,
        file_name="resumen_rutas_cvrp.csv",
        mime="text/csv",
    )

    # Métricas y analíticas visuales de la flota
    st.header("📈 Utilización de la Flota")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_resumen["Vehículo"].astype(str), df_resumen["Utilización (%)"])
    ax.set_title("Nivel de Utilización de Vehículos")
    ax.set_xlabel("Vehículo")
    ax.set_ylabel("Utilización (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Renderizado de Mapas (PyDeck / Nativo)
    st.header("🗺️ Ubicación Geográfica y Rutas")
    mapa_puntos = puntos.rename(columns={"Latitud": "lat", "Longitud": "lon"})

    if pdk is not None:
        segmentos = crear_segmentos_ruta(puntos, rutas)

        layer_puntos = pdk.Layer(
            "ScatterplotLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_radius=700,
            get_fill_color="[40, 120, 200, 180]",
            pickable=True,
        )

        layer_texto = pdk.Layer(
            "TextLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_text="Nombre",
            get_size=14,
            get_color="[0, 0, 0, 255]",
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
        )

        layer_rutas = pdk.Layer(
            "PathLayer",
            data=segmentos,
            get_path="path",
            get_width=5,
            get_color="[220, 80, 60, 200]",
            pickable=True,
        )

        vista = pdk.ViewState(
            latitude=puntos["Latitud"].mean(),
            longitude=puntos["Longitud"].mean(),
            zoom=9,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=vista,
                layers=[layer_rutas, layer_puntos, layer_texto],
                tooltip={"text": "{Nombre}"},
            )
        )
    else:
        st.warning(
            "Para ver las rutas completas en el mapa instala pydeck (`pip install pydeck`). "
            "Mostrando visualización de puntos básica."
        )
        st.map(mapa_puntos[["lat", "lon"]])

    # Conclusiones finales y reportería de rendimiento empresarial
    st.header("📑 Análisis Gerencial")
    vehiculo_max = df_resumen.loc[
        df_resumen["Utilización (%)"].idxmax(), "Vehículo"
    ]
    utilizacion_promedio = round(df_resumen["Utilización (%)"].mean(), 2)

    st.success(
        f"""
        • Se atendió una demanda total de {demanda_total:,} kg.
        • Se utilizaron {len(rutas)} vehículos de {num_vehiculos} disponibles.
        • La distancia total recorrida fue de {distancia_total_km} km.
        • El vehículo con mayor utilización fue el Vehículo {vehiculo_max}.
        • La utilización promedio de la flota fue de {utilizacion_promedio} %.
        • El modelo respetó la restricción de capacidad máxima de {capacidad_vehiculo:,} kg por vehículo.
        • Las rutas fueron calculadas buscando minimizar la distancia total recorrida.
        """
    )

    st.header("✅ Conclusiones")
    st.info(
        f"""
        1. El modelo CVRP permitió optimizar la distribución de alimentos desde el CEDI Tocancipá.
        2. La solución respetó la capacidad máxima de carga de cada vehículo.
        3. La operación atendió {clientes} clientes con una demanda total de {demanda_total:,} kg.
        4. La distancia total estimada fue de {distancia_total_km} km.
        5. La aplicación en Streamlit facilita la visualización de rutas, cargas, utilización de flota y resultados ejecutivos.
        6. Google OR-Tools permitió resolver el problema de ruteo de forma eficiente usando restricciones de capacidad.
        """
    )
