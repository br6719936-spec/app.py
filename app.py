import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ===================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTADOS
# ===================================================
st.set_page_config(
    page_title="Control Tower Logística CVRP Pro", page_icon="🌐", layout="wide"
)

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
# 2. FUNCIONES LÓGICAS Y MATEMÁTICAS
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


def resolver_cvrp_industrial(
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
    factor_trafico,
    factor_emision,
):
    """Resuelve el CVRP avanzado incluyendo sostenibilidad, costos e impacto de tráfico."""
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
        return None, "No se encontró una solución factible con los parámetros actuales."

    rutas = []
    distancia_total_m = 0
    costo_total = 0
    co2_total = 0

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
        distancia_total_m += distancia_ruta

        # Innovación: Tiempos alterados por condiciones de tráfico simuladas
        tiempo_transito_hrs = (distancia_km / velocidad) * factor_trafico
        tiempo_descarga_hrs = (len(clientes_visitados) * tiempo_servicio) / 60
        tiempo_total_ruta = round(tiempo_transito_hrs + tiempo_descarga_hrs, 2)

        # Innovación: Huella Ecológica e Indicadores de Rendimiento de Costos
        costo_ruta = round((distancia_km * costo_km) + costo_fijo, 2)
        co2_ruta = round(distancia_km * factor_emision, 2)
        costo_por_kg = round(costo_ruta / carga, 2) if carga > 0 else 0

        costo_total += costo_ruta
        co2_total += co2_ruta

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
                "Costo/Kg ($)": costo_por_kg,
                "Huella CO2 (kg)": co2_ruta,
            }
        )

    return {
        "rutas": rutas,
        "distancia_total_km": round(distancia_total_m / 1000, 2),
        "costo_total_operacion": round(costo_total, 2),
        "co2_total_kg": round(co2_total, 2),
    }, None


def crear_segmentos_ruta(df, rutas):
    """Genera los segmentos para PyDeck incluyendo colores dinámicos por ID de vehículo."""
    colores_paleta = [
        [230, 57, 70, 220],
        [29, 53, 87, 220],
        [74, 155, 102, 220],
        [241, 146, 14, 220],
        [155, 93, 229, 220],
        [0, 180, 216, 220],
    ]

    segmentos = []
    for index_ruta, ruta in enumerate(rutas):
        indices = ruta["Ruta índices"]
        color_vehiculo = colores_paleta[index_ruta % len(colores_paleta)]

        for origen, destino in zip(indices[:-1], indices[1:]):
            segmentos.append(
                {
                    "Vehículo ID": f"Vehículo {ruta['Vehículo']}",
                    "Color": color_vehiculo,
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
# 3. INTERFAZ DE USUARIO (CONTROL TOWER DESIGN)
# ===================================================
st.title("🌐 Control Tower: Inteligencia de Rutas & Analítica Sostenible")
st.markdown("### Modelo Avanzado de Optimización Logística y Reporte de Emisiones ESG")
st.markdown("---")

# --- SIDEBAR COMPLETO ---
st.sidebar.header("⚙️ Configuración de Flota")
num_vehiculos = st.sidebar.number_input(
    "Vehículos Disponibles", min_value=1, max_value=10, value=4
)
capacidad_vehiculo = st.sidebar.number_input(
    "Capacidad Unitaria (kg)", min_value=100, max_value=10000, value=2200, step=100
)
capacidades = [int(capacidad_vehiculo)] * int(num_vehiculos)

st.sidebar.header("💰 Variables Financieras")
costo_por_km = st.sidebar.number_input(
    "Costo Variable por Km ($)", value=3500.0, step=200.0
)
costo_fijo_vehiculo = st.sidebar.number_input(
    "Costo de Activación de Vehículo ($)", value=80000.0, step=5000.0
)

st.sidebar.header("⏱️ Gestión de Tiempos & Tráfico")
velocidad_promedio = st.sidebar.slider(
    "Velocidad Comercial Base (km/h)", min_value=10, max_value=90, value=40
)
tiempo_servicio_cliente = st.sidebar.number_input(
    "Tiempo de Descarga (minutos)", min_value=0, value=25, step=5
)

# Innovación: Simulación del factor de tráfico en tiempo real
estado_trafico = st.sidebar.select_slider(
    "🚦 Simulación de Estado de Tráfico",
    options=["Fluido (Sin retraso)", "Moderado (+15%)", "Pesado (+40%)"],
    value="Moderado (+15%)",
)
dic_trafico = {
    "Fluido (Sin retraso)": 1.0,
    "Moderado (+15%)": 1.15,
    "Pesado (+40%)": 1.40,
}

st.sidebar.header("🌱 Indicadores de Sostenibilidad (ESG)")
tipo_combustible = st.sidebar.selectbox(
    "Tipo de Vehículo / Combustible",
    ["Camión Diésel Convencional", "Camión Turbo Gasolina", "Vehículo Híbrido"],
)
# Factores reales de emisión aproximados (kg de CO2 por kilómetro)
dic_emisiones = {
    "Camión Diésel Convencional": 0.27,
    "Camión Turbo Gasolina": 0.21,
    "Vehículo Híbrido": 0.12,
}

st.sidebar.header("🔬 Motor de Búsqueda")
tiempo_busqueda = st.sidebar.slider(
    "Tiempo de Cómputo (segundos)", min_value=1, max_value=20, value=5
)

# --- FIN SIDEBAR ---

# Formulario interactivo de demanda
with st.expander("✏️ Registro de Clientes y Demandas en la Sabana de Bogotá"):
    puntos = st.data_editor(
        DATOS_INICIALES, use_container_width=True, num_rows="fixed"
    )

puntos = puntos.reset_index(drop=True)
puntos.loc[DEPOSITO, "Demanda (kg)"] = 0
nombres = puntos["Nombre"].tolist()
demandas = puntos["Demanda (kg)"].astype(int).tolist()

demanda_total = int(sum(demandas))
capacidad_total = int(sum(capacidades))
clientes = len(puntos) - 1

# KPIs operacionales antes de presionar optimizar
col1, col2, col3, col4 = st.columns(4)
col1.metric("`🚛 Flota Disponible`", num_vehiculos)
col2.metric("`📦 Demanda Pendiente`", f"{demanda_total:,} kg")
col3.metric("`🏋️ Capacidad de Carga de Red`", f"{capacidad_total:,} kg")
col4.metric("`📍 Destinos`", clientes)
st.markdown("---")

if demanda_total > capacidad_total:
    st.error(
        "🚨 Capacidad Insuficiente: La demanda supera los límites físicos de la flota configurada."
    )

matriz = construir_matriz_distancias(puntos)

if st.button("🚀 Ejecutar Algoritmo Genético CVRP", disabled=demanda_total > capacidad_total):
    with st.spinner("Computando matrices espaciales y optimizando rutas..."):
        resultado, error = resolver_cvrp_industrial(
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
            factor_trafico=dic_trafico[estado_trafico],
            factor_emision=dic_emisiones[tipo_combustible],
        )

        if error:
            st.error(error)
            st.session_state.optimizado = False
        else:
            st.session_state.resultado = resultado
            st.session_state.optimizado = True

# ===================================================
# 4. DASHBOARD DE RESULTADOS AVANZADOS E INNOVADORES
# ===================================================
if st.session_state.optimizado and st.session_state.resultado:
    rutas = st.session_state.resultado["rutas"]
    distancia_total_km = st.session_state.resultado["distancia_total_km"]
    costo_total_operacion = st.session_state.resultado["costo_total_operacion"]
    co2_total_kg = st.session_state.resultado["co2_total_kg"]

    # --- CUADRO DE MANDOS DE ALTA GERENCIA (KPIs de Innovación) ---
    st.header("📊 Métricas de Control General")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💵 Costo Total de Distribución", f"${costo_total_operacion:,}")
    m2.metric(
        "🌱 Huella de Carbono del Despacho",
        f"{co2_total_kg:,} kg CO₂",
        delta=f"Combustible: {tipo_combustible}",
        delta_color="inverse",
    )
    m3.metric("⏱️ Condición de Tránsito", estado_trafico)
    m4.metric("🚛 Flota Activa en Operación", f"{len(rutas)} Camiones")
    st.markdown("---")

    # --- MAPA CON FILTROS DINÁMICOS (Innovación en visualización) ---
    st.header("🗺️ Torre de Control Mapas e Infraestructura")

    if pdk is not None:
        df_segmentos = crear_segmentos_ruta(puntos, rutas)

        # Innovación: Permitir al usuario aislar vehículos específicos del mapa en tiempo real
        lista_vehiculos = df_segmentos["Vehículo ID"].unique().tolist()
        vehiculos_seleccionados = st.multiselect(
            "🔍 Filtrar visualización de mapa por Vehículo (vacío para ver todos)",
            options=lista_vehiculos,
            default=lista_vehiculos,
        )

        df_filtrado_segmentos = df_segmentos[
            df_segmentos["Vehículo ID"].isin(vehiculos_seleccionados)
        ]

        layer_puntos = pdk.Layer(
            "ScatterplotLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_radius=600,
            get_fill_color="[30, 41, 59, 200]",
            pickable=True,
        )
        layer_texto = pdk.Layer(
            "TextLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_text="Nombre",
            get_size=13,
            get_color="[15, 23, 42, 255]",
            get_alignment_baseline="'bottom'",
        )
        layer_rutas = pdk.Layer(
            "PathLayer",
            data=df_filtrado_segmentos,
            get_path="path",
            get_width=5.5,
            get_color="Color",
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
        st.map(puntos.rename(columns={"Latitud": "lat", "Longitud": "lon"}))

    # --- TABLA DE REPORTES FINANCIEROS Y OPERATIVOS ---
    st.header("📋 Analítica Detallada por Ruta de Distribución")
    df_resumen = pd.DataFrame(rutas)
    df_tabla = df_resumen[
        [
            "Vehículo",
            "Carga (kg)",
            "Distancia (km)",
            "Utilización (%)",
            "Tiempo Estimado (horas)",
            "Costo Ruta ($)",
            "Costo/Kg ($)",
            "Huella CO2 (kg)",
            "Ruta",
        ]
    ]

    st.dataframe(df_tabla, use_container_width=True)

    st.download_button(
        label="⬇️ Descargar Reporte Financiero-Ambiental (CSV)",
        data=df_tabla.to_csv(index=False).encode("utf-8"),
        file_name="reporte_logistica_integral.csv",
        mime="text/csv",
    )

    # --- GRÁFICOS CRUZADOS DE RENDIMIENTO ---
    st.header("📈 Gráficos de Eficiencia Operativa")
    g1, g2 = st.columns(2)

    with g1:
        # Gráfico de costo financiero vs huella ambiental
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.bar(
            df_resumen["Vehículo"].astype(str),
            df_resumen["Costo Ruta ($)"],
            color="#E63946",
            alpha=0.7,
            label="Costo ($)",
        )
        ax1.set_ylabel("Costo ($)", color="#E63946")
        ax1.set_xlabel("Vehículos")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            df_resumen["Vehículo"].astype(str),
            df_resumen["Huella CO2 (kg)"],
            color="#4A9B66",
            marker="s",
            linewidth=2,
            label="CO₂ (kg)",
        )
        ax1_twin.set_ylabel("Huella CO2 (kg)", color="#4A9B66")
        plt.title("Relación Costo vs Huella de Carbono por Unidad")
        st.pyplot(fig1)

    with g2:
        # Gráfico del costo unitario por kilogramo transportado
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(
            df_resumen["Vehículo"].astype(str),
            df_resumen["Costo/Kg ($)"],
            color="#1D3557",
            alpha=0.8,
        )
        ax2.set_ylabel("Costo por Kg ($/Kg)")
        ax2.set_xlabel("Vehículos")
        plt.title("Costo de Distribución Unitario (Eficiencia por Kg)")
        st.pyplot(fig2)
