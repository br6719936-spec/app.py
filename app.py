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
    page_title="Dashboard Logístico CVRP Avanzado", page_icon="🚚", layout="wide"
)

# Inicializar estados de sesión para el optimizador
if "optimizado" not in st.session_state:
    st.session_state.optimizado = False
if "resultado" not in st.session_state:
    st.session_state.resultado = None

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
# 2. FUNCIONES LÓGICAS Y DE OPTIMIZACIÓN
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


def resolver_cvrp_avanzado(
    matriz,
    demandas,
    capacidades,
    deposito,
    segundos,
    nombres,
    costo_km,
    costo_fijo,
    velocidad,
    tiempo_servicio,
):
    """Resuelve el problema CVRP incorporando costos y tiempos."""
    if sum(demandas) > sum(capacidades):
        return None, "La demanda total supera la capacidad total disponible."

    manager = pywrapcp.RoutingIndexManager(
        len(matriz), len(capacidades), deposito
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return matriz[manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)
        ]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Incorporar Costo Fijo por activar cada vehículo
    for i in range(len(capacidades)):
        routing.SetFixedCostOfVehicle(int(costo_fijo), i)

    def demand_callback(from_index):
        return demandas[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, capacidades, True, "Capacidad"
    )

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

    rutas = []
    distancia_total = 0
    costo_operativo_total = 0

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

        distancia_km = round(distancia_ruta / 1000, 2)
        distancia_total += distancia_ruta

        # Cálculo estimado de tiempos (Tránsito + Descargas)
        tiempo_transito_hrs = distancia_km / velocidad
        tiempo_descarga_hrs = (len(clientes_visitados) * tiempo_servicio) / 60
        tiempo_total_ruta = round(tiempo_transito_hrs + tiempo_descarga_hrs, 2)

        # Cálculo financiero por ruta
        costo_ruta = round((distancia_km * costo_km) + costo_fijo, 2)
        costo_operativo_total += costo_ruta

        rutas.append(
            {
                "Vehículo": vehicle_id + 1,
                "Ruta índices": ruta_indices,
                "Ruta": " ➜ ".join([nombres[i] for i in ruta_indices]),
                "Carga (kg)": carga,
                "Distancia (km)": distancia_km,
                "Utilización (%)": round(
                    (carga / capacidades[vehicle_id]) * 100, 2
                ),
                "Tiempo Estimado (horas)": tiempo_total_ruta,
                "Costo Ruta ($)": costo_ruta,
            }
        )

    return {
        "rutas": rutas,
        "distancia_total_km": round(distancia_total / 1000, 2),
        "costo_total_operacion": round(costo_operativo_total, 2),
    }, None


def crear_segmentos_ruta(df, rutas):
    """Genera los segmentos espaciales para la visualización en PyDeck con colores por vehículo."""
    colores_paleta = [
        [230, 57, 70, 220],    # Vehículo 1: Rojo
        [29, 53, 87, 220],     # Vehículo 2: Azul Oscuro
        [74, 155, 102, 220],   # Vehículo 3: Verde
        [241, 146, 14, 220],   # Vehículo 4: Naranja
        [155, 93, 229, 220],   # Vehículo 5: Morado
        [0, 180, 216, 220]     # Vehículo 6: Celeste
    ]
    
    segmentos = []
    for index_ruta, ruta in enumerate(rutas):
        indices = ruta["Ruta índices"]
        color_vehiculo = colores_paleta[index_ruta % len(colores_paleta)]
        
        for origen, destino in zip(indices[:-1], indices[1:]):
            segmentos.append(
                {
                    "Vehículo": f"Vehículo {ruta['Vehículo']}",
                    "Color": color_vehiculo,
                    "path": [
                        [df.loc[origen, "Longitud"], df.loc[origen, "Latitud"]],
                        [df.loc[destino, "Longitud"], df.loc[destino, "Latitud"]],
                    ],
                }
            )
    return pd.DataFrame(segmentos)


# ===================================================
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# ===================================================
st.title("🚚 Dashboard Inteligente de Optimización Logística Avanzada")
st.markdown("### Caso de Estudio: Distribución de Alimentos en la Sabana de Bogotá")
st.markdown("---")

# --- PANEL LATERAL CON PARÁMETROS ENRIQUECIDOS ---
st.sidebar.header("⚙️ Parámetros de la Flota")
num_vehiculos = st.sidebar.number_input(
    "Número de vehículos disponibles", min_value=1, max_value=10, value=4, step=1
)
capacidad_vehiculo = st.sidebar.number_input(
    "Capacidad por vehículo (kg)",
    min_value=100,
    max_value=10000,
    value=2200,
    step=100,
)

st.sidebar.header("💰 Parámetros Financieros")
costo_por_km = st.sidebar.number_input(
    "Costo variable por kilómetro ($/km)", min_value=0.0, value=3500.0, step=500.0
)
costo_fijo_vehiculo = st.sidebar.number_input(
    "Costo fijo por activar vehículo ($)", min_value=0.0, value=80000.0, step=5000.0
)

st.sidebar.header("⏱️ Parámetros Operativos (Tiempos)")
velocidad_promedio = st.sidebar.slider(
    "Velocidad promedio de tránsito (km/h)", min_value=10, max_value=90, value=40
)
tiempo_servicio_cliente = st.sidebar.number_input(
    "Tiempo de descarga por cliente (minutos)", min_value=0, value=25, step=5
)

st.sidebar.header("🚀 Configuración del Motor")
tiempo_busqueda = st.sidebar.slider(
    "Tiempo límite de optimización (segundos)", min_value=1, max_value=30, value=5
)

capacidades = [int(capacidad_vehiculo)] * int(num_vehiculos)

# Editor de datos de clientes
with st.expander("✏️ Editar datos de clientes y demandas"):
    puntos = st.data_editor(
        DATOS_INICIALES, use_container_width=True, num_rows="fixed"
    )

puntos = puntos.reset_index(drop=True)
puntos.loc[DEPOSITO, "Demanda (kg)"] = 0
nombres = puntos["Nombre"].tolist()
demandas = puntos["Demanda (kg)"].astype(int).tolist()

# Indicadores globales previos
demanda_total = int(sum(demandas))
capacidad_total = int(sum(capacidades))
clientes = len(puntos) - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("`🚛 Flota Máxima`", num_vehiculos)
col2.metric("`📦 Demanda Requerida`", f"{demanda_total:,} kg")
col3.metric("`🏋️ Capacidad Máxima Flota`", f"{capacidad_total:,} kg")
col4.metric("`📍 Puntos de Entrega`", clientes)
st.markdown("---")

if demanda_total > capacidad_total:
    st.error(
        "⚠️ Alerta Operativa: La demanda total supera la capacidad máxima de tu flota actual."
    )

# Matriz de distancias
matriz = construir_matriz_distancias(puntos)

# Botón ejecutor
if st.button("🚀 Optimizar Operación Logística", disabled=demanda_total > capacidad_total):
    with st.spinner("Buscando las mejores rutas de mínimo costo..."):
        resultado, error = resolver_cvrp_avanzado(
            matriz=matriz,
            demandas=demandas,
            capacidades=capacidades,
            deposito=DEPOSITO,
            segundos=tiempo_busqueda,
            nombres=nombres,
            costo_km=costo_por_km,
            costo_fijo=costo_fijo_vehiculo,
            velocidad=velocidad_promedio,
            tiempo_servicio=tiempo_servicio_cliente,
        )

        if error:
            st.error(error)
            st.session_state.optimizado = False
        else:
            st.session_state.resultado = resultado
            st.session_state.optimizado = True

# ===================================================
# 4. VISUALIZACIÓN DE RESULTADOS AVANZADOS
# ===================================================
if st.session_state.optimizado and st.session_state.resultado:
    rutas = st.session_state.resultado["rutas"]
    distancia_total_km = st.session_state.resultado["distancia_total_km"]
    costo_total_operacion = st.session_state.resultado["costo_total_operacion"]

    st.header("📋 Desglose Técnico por Ruta Activa")
    
    for r in rutas:
        with st.container():
            st.subheader(f"🚛 Ruta Asignada al Vehículo {r['Vehículo']}")
            c_it1, c_it2, c_it3, c_it4 = st.columns(4)
            c_it1.write(f"📦 **Carga:** {r['Carga (kg)']} / {capacidad_vehiculo} kg ({r['Utilización (%)']}%)")
            c_it2.write(f"📍 **Distancia:** {r['Distancia (km)']} km")
            c_it3.write(f"⏱️ **Duración:** {r['Tiempo Estimado (horas)']} hrs")
            c_it4.write(f"💰 **Costo de Ruta:** ${r['Costo Ruta ($)']:,}")
            st.caption(f"**Secuencia óptima:** {r['Ruta']}")
            st.markdown("---")

    # Resumen y cuadro ejecutivo financiero
    st.header("📊 Cuadro de Mando Financiero y Ejecutivo")
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("✅ Distancia Consolidada", f"{distancia_total_km} km")
    c_m2.metric("💵 Costo Operativo Total", f"${costo_total_operacion:,}")
    c_m3.metric("🚚 Camiones Utilizados", f"{len(rutas)} de {num_vehiculos}")

    df_resumen = pd.DataFrame(rutas)
    df_tabla = df_resumen[
        ["Vehículo", "Carga (kg)", "Distancia (km)", "Utilización (%)", "Tiempo Estimado (horas)", "Costo Ruta ($)", "Ruta"]
    ]
    st.dataframe(df_tabla, use_container_width=True)

    st.download_button(
        label="⬇️ Exportar Manifiesto de Carga (CSV)",
        data=df_tabla.to_csv(index=False).encode("utf-8"),
        file_name="manifiesto_rutas_avanzado.csv",
        mime="text/csv",
    )

    # Gráfico comparativo de costos vs utilización
    st.header("📈 Eficiencia y Costos por Vehículo")
    fig, ax1 = plt.subplots(figsize=(8, 3.5))

    ax2 = ax1.twinx()
    ax1.bar(df_resumen["Vehículo"].astype(str), df_resumen["Costo Ruta ($)"], color="g", alpha=0.6, label="Costo ($)")
    ax2.plot(df_resumen["Vehículo"].astype(str), df_resumen["Utilización (%)"], color="b", marker="o", linewidth=2, label="Utilización %")

    ax1.set_xlabel("Vehículo")
    ax1.set_ylabel("Costo de Operación ($)", color="g")
    ax2.set_ylabel("Nivel de Utilización (%)", color="b")
    ax2.set_ylim(0, 110)
    plt.title("Análisis de Costo Financiero frente a la Capacidad Utilizada")
    
    st.pyplot(fig)

    # Mapa con Trazados de Líneas de Diferentes Colores
    st.header("🗺️ Trazado de Rutas Independientes en el Mapa")
    mapa_puntos = puntos.rename(columns={"Latitud": "lat", "Longitud": "lon"})

    if pdk is not None:
        segmentos = crear_segmentos_ruta(puntos, rutas)

        layer_puntos = pdk.Layer(
            "ScatterplotLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_radius=600,
            get_fill_color="[34, 139, 34, 200]",
            pickable=True,
        )
        layer_texto = pdk.Layer(
            "TextLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_text="Nombre",
            get_size=13,
            get_color="[40, 40, 40, 255]",
            get_alignment_baseline="'bottom'",
        )
        layer_rutas = pdk.Layer(
            "PathLayer",
            data=segmentos,
            get_path="path",
            get_width=5,
            get_color="Color",  # <--- Utiliza la columna dinámica de color por vehículo
            pickable=True,
        )
        
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=puntos["Latitud"].mean(),
                    longitude=puntos["Longitud"].mean(),
                    zoom=9.2,
                ),
                layers=[layer_rutas, layer_puntos, layer_texto],
                tooltip={"text": "{Nombre}"},
            )
        )
    else:
        st.warning("PyDeck no está disponible. Mostrando mapa base predeterminado.")
        st.map(mapa_puntos[["lat", "lon"]])
