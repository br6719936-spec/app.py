import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Intentar importar pydeck de forma segura para evitar caídas del sistema
try:
    import pydeck as pdk
except ImportError:
    pdk = None

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ===================================================
# 1. CONFIGURACIÓN DEL SISTEMA Y ESTADOS
# ===================================================
st.set_page_config(
    page_title="Control Tower Logística CVRP Enterprise", page_icon="🌐", layout="wide"
)

# Control de persistencia de datos en memoria para evitar reseteos al interactuar
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
# 2. CAPA DE CÓMPUTO LOGÍSTICO Y OPTIMIZACIÓN
# ===================================================
def distancia_haversine(coord1, coord2):
    """Calcula la distancia real esférica aproximada entre dos coordenadas."""
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
    """Construye la matriz de distancias en metros punto a punto."""
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


def resolver_cvrp_enterprise(
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
    factor_clima,
    factor_emision,
    rendimiento_base,
):
    """Algoritmo OR-Tools CVRP con inyección de variables financieras, logísticas y ESG."""
    if sum(demandas) > sum(capacidades):
        return None, "Error: La demanda agregada excede la capacidad instalada de la flota."

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

    # Penalización / Costo por activar cada camión
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

    # Parámetros avanzados del solucionador heurístico
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
        return None, "No se encontró una solución viable con las restricciones vigentes."

    rutas = []
    distancia_total_m = 0
    costo_total_operacion = 0
    co2_total_kg = 0
    combustible_total_gal = 0

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

        # Descartar vehículos no utilizados por el algoritmo
        if not clientes_visitados:
            continue

        distancia_km = round(distancia_ruta / 1000, 2)
        distancia_total_m += distancia_ruta

        # Impacto matemático combinatorio de Clima + Tráfico en el tiempo de tránsito
        velocidad_afectada = velocidad * (1 / factor_trafico) * (1 / factor_clima)
        tiempo_transito_hrs = distancia_km / velocidad_afectada
        tiempo_descarga_hrs = (len(clientes_visitados) * tiempo_servicio) / 60
        tiempo_total_ruta = round(tiempo_transito_hrs + tiempo_descarga_hrs, 2)

        # Telemetría de combustible y costos financieros cruzados
        consumo_galones = round(distancia_km / rendimiento_base, 2)
        costo_ruta = round((distancia_km * costo_km) + costo_fijo, 2)
        co2_ruta = round(distancia_km * factor_emision, 2)
        costo_por_kg = round(costo_ruta / carga, 2) if carga > 0 else 0
        utilizacion_porcentaje = round((carga / capacidades[vehicle_id]) * 100, 2)

        costo_total_operacion += costo_ruta
        co2_total_kg += co2_ruta
        combustible_total_gal += consumo_galones

        rutas.append(
            {
                "Vehículo": vehicle_id + 1,
                "Ruta índices": ruta_indices,
                "Ruta": " ➜ ".join([nombres[i] for i in ruta_indices]),
                "Carga (kg)": carga,
                "Distancia (km)": distancia_km,
                "Utilización (%)": utilizacion_porcentaje,
                "Tiempo Estimado (horas)": tiempo_total_ruta,
                "Consumo (Gal)": consumo_galones,
                "Costo Ruta ($)": costo_ruta,
                "Costo/Kg ($)": costo_por_kg,
                "Huella CO2 (kg)": co2_ruta,
            }
        )

    return {
        "rutas": rutas,
        "distancia_total_km": round(distancia_total_m / 1000, 2),
        "costo_total_operacion": round(costo_total_operacion, 2),
        "co2_total_kg": round(co2_total_kg, 2),
        "combustible_total_gal": round(combustible_total_gal, 2),
    }, None


def crear_segmentos_ruta(df, rutas):
    """Estructura las coordenadas de vectores espaciales de cada vehículo con paleta cromática diferenciada."""
    colores_paleta = [
        [230, 57, 70, 220],    # Vehículo 1: Rojo Coral
        [29, 53, 87, 220],     # Vehículo 2: Azul Industrial
        [74, 155, 102, 220],   # Vehículo 3: Verde Esmeralda
        [241, 146, 14, 220],   # Vehículo 4: Naranja Tráfico
        [155, 93, 229, 220],   # Vehículo 5: Púrpura Eléctrico
        [0, 180, 216, 220],    # Vehículo 6: Cyan Cobalto
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
                        [df.loc[destino, "Longitud"], df.loc[destino, "Latitud"]],
                    ],
                }
            )
    return pd.DataFrame(segmentos)


# ===================================================
# 3. PANELES DE CONTROL (INTERFAZ GRÁFICA)
# ===================================================
st.title("🌐 Enterprise Control Tower: Logística & Analítica de Distribución")
st.markdown("### Centro de Simulación Avanzada de Carga, Costos Operativos y Huella de Carbono (ESG)")
st.markdown("---")

# --- BLOQUE SIDEBAR CON PARÁMETROS INDUSTRIALES ---
st.sidebar.header("⚙️ Configuración de Flota")
num_vehiculos = st.sidebar.number_input(
    "Capacidad de Flota (Vehículos)", min_value=1, max_value=10, value=4
)
capacidad_vehiculo = st.sidebar.number_input(
    "Capacidad Máxima Unitaria (kg)", min_value=100, max_value=10000, value=2200, step=100
)
capacidades = [int(capacidad_vehiculo)] * int(num_vehiculos)

st.sidebar.header("💰 Modelado Financiero")
costo_por_km = st.sidebar.number_input(
    "Costo Variable por Kilómetro ($/Km)", value=3500.0, step=200.0
)
costo_fijo_vehiculo = st.sidebar.number_input(
    "Costo de Despacho / Fijo de Activación ($)", value=80000.0, step=5000.0
)

st.sidebar.header("⏱️ Gestión de Tiempos e Imprevistos")
velocidad_promedio = st.sidebar.slider(
    "Velocidad Comercial Base (km/h)", min_value=10, max_value=90, value=40
)
tiempo_servicio_cliente = st.sidebar.number_input(
    "Tiempo de Servicio de Descarga (min)", min_value=0, value=25, step=5
)

# Factores exógenos de tráfico y clima
estado_trafico = st.sidebar.select_slider(
    "🚦 Estado de Flujo Vial (Tráfico)",
    options=["Fluido (1.0x)", "Moderado (1.15x)", "Congestionado (1.45x)"],
    value="Moderado (1.15x)",
)
dic_trafico = {"Fluido (1.0x)": 1.0, "Moderado (1.15x)": 1.15, "Congestionado (1.45x)": 1.45}

estado_clima = st.sidebar.select_slider(
    "🌧️ Condiciones Meteorológicas",
    options=["Despejado (1.0x)", "Lluvia Ligera (1.10x)", "Tormenta / Niebla (1.30x)"],
    value="Despejado (1.0x)",
)
dic_clima = {"Despejado (1.0x)": 1.0, "Lluvia Ligera (1.10x)": 1.10, "Tormenta / Niebla (1.30x)": 1.30}

st.sidebar.header("🌱 Matriz de Eficiencia Sostenible (ESG)")
tipo_combustible = st.sidebar.selectbox(
    "Tipología de Motor de Flota",
    ["Camión Diésel Pesado", "Camión Turbo Gasolina", "Vehículo Híbrido Logístico"],
)
dic_emisiones = {"Camión Diésel Pesado": 0.27, "Camión Turbo Gasolina": 0.21, "Vehículo Híbrido Logístico": 0.12}
dic_rendimiento = {"Camión Diésel Pesado": 18.0, "Camión Turbo Gasolina": 14.0, "Vehículo Híbrido Logístico": 25.0}

st.sidebar.header("🔬 Algoritmo")
tiempo_busqueda = st.sidebar.slider(
    "Tiempo de Cómputo Metaheurístico (s)", min_value=1, max_value=20, value=5
)
# --- FIN SIDEBAR ---


# Panel del editor de datos de la red
with st.expander("✏️ Maestro de Destinos y Demandas Comerciales (Edición en vivo)"):
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

# KPIs operacionales base preliminares
col1, col2, col3, col4 = st.columns(4)
col1.metric("`🚛 Capacidad Flota`", f"{num_vehiculos} Unidades")
col2.metric("`📦 Demanda Global Requerida`", f"{demanda_total:,} kg")
col3.metric("`🏋️ Capacidad Máxima de Red`", f"{capacidad_total:,} kg")
col4.metric("`📍 Clientes Maestro`", clientes)
st.markdown("---")

if demanda_total > capacidad_total:
    st.error("🚨 Ruptura de Restricción: La demanda excede los límites físicos de carga de la flota configurada.")

matriz = construir_matriz_distancias(puntos)

# Disparador del motor de optimización de Google OR-Tools
if st.button("🚀 Ejecutar Optimización Integral de Flota", disabled=demanda_total > capacidad_total):
    with st.spinner("Procesando combinatoria espacial de mínimo costo..."):
        resultado, error = resolver_cvrp_enterprise(
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
            factor_clima=dic_clima[estado_clima],
            factor_emision=dic_emisiones[tipo_combustible],
            rendimiento_base=dic_rendimiento[tipo_combustible],
        )

        if error:
            st.error(error)
            st.session_state.optimizado = False
        else:
            st.session_state.resultado = resultado
            st.session_state.optimizado = True


# ===================================================
# 4. CAPA DE INTERFACES INTERACTIVAS Y GRÁFICAS (POST-PROCESAMIENTO)
# ===================================================
if st.session_state.optimizado and st.session_state.resultado:
    rutas = st.session_state.resultado["rutas"]
    distancia_total_km = st.session_state.resultado["distancia_total_km"]
    costo_total_operacion = st.session_state.resultado["costo_total_operacion"]
    co2_total_kg = st.session_state.resultado["co2_total_kg"]
    combustible_total_gal = st.session_state.resultado["combustible_total_gal"]

    # --- MÉTRICAS DE CONTROL DE ALTA GERENCIA ---
    st.header("📊 Tablero de Control y Viabilidad Financiera")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💵 Costo de Operación Total", f"${costo_total_operacion:,}")
    m2.metric("🌱 Impacto CO2 Generado", f"{co2_total_kg:,} kg CO₂", delta=tipo_combustible, delta_color="inverse")
    m3.metric("⛽ Combustible Estimado", f"{combustible_total_gal:,} Gal")
    m4.metric("🛣️ Kilómetros Consolidados", f"{distancia_total_km:,} Km")
    st.markdown("---")

    # --- MAPA CON FILTROS DE RUTA AVANZADOS (TEMA VIAL NORMAL) ---
    st.header("🗺️ Monitoreo Espacial de Rutas por Vehículo")

    if pdk is not None:
        df_segmentos = crear_segmentos_ruta(puntos, rutas)

        # Filtro dinámico e interactivo multiselect
        lista_vehiculos = df_segmentos["Vehículo ID"].unique().tolist()
        vehiculos_seleccionados = st.multiselect(
            "🔍 Filtrar visualización en tiempo real (Deja vacío para desplegar toda la flota):",
            options=lista_vehiculos,
            default=lista_vehiculos,
        )

        if not vehiculos_seleccionados:
            df_filtrado_segmentos = df_segmentos
        else:
            df_filtrado_segmentos = df_segmentos[df_segmentos["Vehículo ID"].isin(vehiculos_seleccionados)]

        layer_puntos = pdk.Layer(
            "ScatterplotLayer",
            data=puntos,
            get_position="[Longitud, Latitud]",
            get_radius=550,
            get_fill_color="[15, 23, 42, 220]",
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
            get_width=6,
            get_color="Color",
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                # Utiliza el estilo ROAD nativo para mostrar el mapa a color convencional
                map_style=pdk.map_styles.ROAD,
                initial_view_state=pdk.ViewState(
                    latitude=puntos["Latitud"].mean(),
                    longitude=puntos["Longitud"].mean(),
                    zoom=9.2,
                ),
                layers=[layer_rutas, layer_puntos, layer_texto],
                tooltip={"text": "{Vehículo ID} pasando por el nodo"},
            )
        )
    else:
        st.map(puntos.rename(columns={"Latitud": "lat", "Longitud": "lon"}))

    # --- REPORTES ANALÍTICOS DE TABLA ---
    st.header("📋 Matriz Completa de Despachos y Telemetría")
    df_resumen = pd.DataFrame(rutas)
    df_tabla = df_resumen[
        [
            "Vehículo",
            "Carga (kg)",
            "Distancia (km)",
            "Utilización (%)",
            "Tiempo Estimado (horas)",
            "Consumo (Gal)",
            "Costo Ruta ($)",
            "Costo/Kg ($)",
            "Huella CO2 (kg)",
            "Ruta",
        ]
    ]
    st.dataframe(df_tabla, use_container_width=True)

    st.download_button(
        label="⬇️ Exportar Libro Técnico de Rutas (CSV)",
        data=df_tabla.to_csv(index=False).encode("utf-8"),
        file_name="manifiesto_enterprise_logistica.csv",
        mime="text/csv",
    )

    # --- SECCIÓN GRÁFICA TRIPLE DE EFICIENCIA LOGÍSTICA ---
    st.header("📈 Business Intelligence: Análisis de Eficiencia Cruzada")
    g1, g2, g3 = st.columns(3)

    with g1:
        # Gráfica 1: Costo vs Sustentabilidad Ambiental
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.bar(df_resumen["Vehículo"].astype(str), df_resumen["Costo Ruta ($)"], color="#E63946", alpha=0.7)
        ax1.set_ylabel("Costo de Operación ($)", color="#E63946")
        ax1.set_xlabel("Vehículo")
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df_resumen["Vehículo"].astype(str), df_resumen["Huella CO2 (kg)"], color="#4A9B66", marker="o", linewidth=2)
        ax1_twin.set_ylabel("Huella CO2 (kg)", color="#4A9B66")
        plt.title("Ecuación de Costo de Ruta vs. Impacto CO₂")
        st.pyplot(fig1)

    with g2:
        # Gráfica 2: Costo Unitario de Transporte por Kilogramo
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.bar(df_resumen["Vehículo"].astype(str), df_resumen["Costo/Kg ($)"], color="#1D3557", alpha=0.8)
        ax2.set_ylabel("Eficiencia Unitario ($ / Kg)")
        ax2.set_xlabel("Vehículo")
        plt.title("Costo Específico por Kilogramo Distribuido")
        st.pyplot(fig2)

    with g3:
        # Gráfica 3: Nivel de Ocupación Volumétrica por Camión
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.bar(df_resumen["Vehículo"].astype(str), df_resumen["Utilización (%)"], color="#F1920E", alpha=0.7)
        ax3.set_ylabel("Porcentaje de Capacidad (%)")
        ax3.set_xlabel("Vehículo")
        ax3.axhline(100, color="red", linestyle="--", alpha=0.5, label="Límite Físico")
        ax3.set_ylim(0, 115)
        plt.title("Nivel de Utilización de Capacidad Útil")
        st.pyplot(fig3)
