import streamlit as st
import pandas as pd
import numpy as np
import pandasql as psql
from torch.nn import HuberLoss
import plotly.graph_objects as go
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
set_random_seed(42)

def crear_placeholder():
    placeholder = st.sidebar.empty()  # Crear un placeholder vacío
    return placeholder

@st.cache_data
def carga_archivos(archivo):
    required_columns = ["TIPO", "PLANTA", "VERSION", "FECHA", "H01", "H02", "H03", "H04", "H05",
                        "H06", "H07", "H08", "H09", "H10", "H11", "H12", "H13", "H14", "H15",
                        "H16","H17", "H18", "H19", "H20", "H21", "H22", "H23", "H24"]
    DB = ""
    alerta = crear_placeholder()
    if archivo is not None:
        df = pd.read_csv(archivo)
        # Validar columnas
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Mostrar advertencia si faltan columnas
            alerta.warning(f":no_entry: El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
            return pd.DataFrame()  # Devuelve un DataFrame vacío si hay columnas faltantes
        else:
            # Si todas las columnas están presentes, eliminar "TIPO" y "VERSION"
            DB = df.drop(columns=['TIPO', 'VERSION'], errors='ignore')
            #print("Base de datos cargada")
            alerta.success("🗃️ :heavy_check_mark: Base de datos cargada")
    else:
        return pd.DataFrame()

    return DB

def obtener_PCH_data(datos):
    query = """
    SELECT strftime('%Y-%m', fecha) AS PERIODO,PLANTA,
    SUM(H01 + H02 + H03 + H04 + H05 + H06 + H07 + H08 + H09 + H10 + H11 + H12 + H13 + H14 + H15 + H16 + H17 + H18 + H19 + H20 + H21 + H22 + H23 + H24)/1e6 AS POT
    FROM datos
    GROUP BY PLANTA, PERIODO
    ORDER BY PLANTA ASC, PERIODO ASC;
    """
    result = psql.sqldf(query, locals())
    result['PERIODO'] = pd.to_datetime(result['PERIODO'])
    return result

def imputar_TS(df, columna_valores):
    umbral = df.loc[:,columna_valores].max() * (15 / 100)
    df.loc[:,columna_valores] = df.loc[:,columna_valores].apply(lambda x: x if x >= umbral else float('nan'))
    df.loc[:,columna_valores] = df.loc[:,columna_valores].interpolate(method='cubic')
    return df

def PCH_preprocess(datos,fecha):
    start_date,end_date = datos['PERIODO'].min(),datos['PERIODO'].max()
    all_periods = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    # Crear un DataFrame con todas las combinaciones posibles de PLANTA y PERIODO
    all_plants = datos['PLANTA'].unique()
    combinations = pd.MultiIndex.from_product([all_plants, all_periods], names=['PLANTA', 'PERIODO'])
    full_df = pd.DataFrame(index=combinations).reset_index()
    full_df['PERIODO'] = pd.to_datetime(full_df['PERIODO'] + '-01')
    # Combinar con el DataFrame original y llenar valores NaN donde faltan datos
    full_df = pd.merge(full_df, datos, how='left', on=['PLANTA', 'PERIODO'])[full_df['PERIODO'] >= fecha].reset_index(drop=True)
    # Combinar resultados VNTA y VNTB
    full_df.loc[full_df['PLANTA'].isin(['VNTA', 'VNTB']), 'PLANTA'] = 'VNT1'
    PCH_data = full_df.groupby(['PLANTA', 'PERIODO'], as_index=False)['POT'].sum(min_count=1)
    PCH_data = PCH_data[PCH_data['PLANTA'] != '2U1G']
    PCH_data = agrupar_zona_PCH(PCH_data,'CAUC',['MND1','SLV1','FLRD','OVJ1','INZ1','ASN1','LPLO'])
    PCH_data = agrupar_zona_PCH(PCH_data,'HTOL',['VNT1','STG1','RCIO','MIR1','PST1'])
    PCH_data = agrupar_zona_PCH(PCH_data,'VAT1',['CAUC','HTOL'])
    return PCH_data

def agrupar_zona_PCH(df, zona, plantas):
    df_grupo = (
      df[df["PLANTA"].isin(plantas)]
      .groupby("PERIODO", as_index=False)
      .agg({"POT": "sum"}))
    df_grupo["PLANTA"] = str(zona)
    df_tot = pd.concat([df, df_grupo], ignore_index=True)
    return df_tot

def cut_data(datos,planta):
    cuts = {
        'RCIO': {'2020-09-01'}
    }
    fecha_corte = next(iter(cuts.get(planta, {'1900-01-01'})))  # Extrae la fecha del conjunto
    datos = datos[datos.PERIODO >= fecha_corte]
    return datos

def setpoint(planta, horizonte):
    base_params = {
        'n_changepoints': 24, 'changepoints_range': 0.9, 'growth': 'discontinuous', 'optimizer': 'AdamW',
        'epochs': 300, 'n_forecasts': horizonte, 'quantiles': []#[0.1, 0.9]
    }
    specific_params = {
        'OVJ1': {'valid_p':0.1},
        'FLRD': {'n_changepoints': 10, 'changepoints_range': 0.8, 'growth': 'linear', 'valid_p':0.1},
        'MIR1': {'n_changepoints': 6, 'growth': 'linear', 'changepoints_range': 0.8, 'valid_p':0.2},
        'INZ1': {'n_changepoints': 24, 'changepoints_range': 0.8, 'loss_func':HuberLoss, 'valid_p':0.1},
        'RCIO': {'n_changepoints': 15, 'changepoints_range': 0.9, 'valid_p':0.1},
        'PST1': {'n_changepoints': 36, 'loss_func':HuberLoss, 'valid_p':0.2},
        'VNT1': {'n_changepoints': 10, 'changepoints_range': 0.8, 'growth': 'linear', 'valid_p':0.1},
        'STG1': {'n_changepoints': 6, 'changepoints_range': 0.9, 'loss_func':HuberLoss, 'growth': 'linear', 'valid_p':0.1},
        'SJN1': {'n_changepoints': 24, 'changepoints_range': 0.9, 'loss_func':HuberLoss, 'valid_p':0.09},
        'ASN1': {'n_changepoints': 5, 'changepoints_range': 0.9, 'loss_func':HuberLoss, 'valid_p':0.2},
        'LPLO': {'n_changepoints': 10, 'loss_func':HuberLoss, 'changepoints_range': 0.75, 'growth': 'linear', 'valid_p':0.1},
        'MND1': {'n_changepoints': 3, 'changepoints_range': 0.9, 'loss_func':HuberLoss, 'valid_p':0.1},
        'SLV1': {'n_changepoints': 13, 'changepoints_range': 0.9, 'loss_func':HuberLoss, 'growth': 'linear', 'valid_p':0.15},
        'CAUC': {'n_changepoints': 10, 'changepoints_range': 0.8, 'loss_func':HuberLoss, 'growth': 'linear', 'valid_p':0.1},
        'HTOL': {'n_changepoints': 6, 'changepoints_range': 0.8, 'loss_func':HuberLoss, 'growth': 'linear', 'valid_p':0.2},
        'VAT1': {'n_changepoints': 10, 'changepoints_range': 0.8, 'loss_func':HuberLoss, 'growth': 'linear', 'valid_p':0.1}
    }
    params = {**base_params, **specific_params.get(planta, {})}

    return params

def entrenar(datos,fecha,horizonte):
    params = setpoint(datos['PLANTA'].unique()[0],horizonte)
    datos = cut_data(datos,datos['PLANTA'].unique()[0])
    datos.drop(["PLANTA"], axis=1, inplace=True)
    datos.columns = ['ds', 'y']
    train,test = datos[datos.ds <= fecha], datos[datos.ds > fecha]
    valid_p = params.pop('valid_p')
    m = NeuralProphet(**params)
    m.add_country_holidays("CO")
    train, val = m.split_df(datos, freq='M', valid_p=valid_p)
    m.fit(train, freq = "M", validation_df=val, early_stopping=True, checkpointing=True)
    std = interv_pron(m,train)
    return m, val, params.get('quantiles'), std

def interv_pron(m,train):
    hist_fitting = m.predict(train)
    errors = hist_fitting['yhat1'].values - train['y'].values
    std_error = np.std(errors)
    return std_error

def pronostico(val,modelo,horizonte,real_lim,std):
    df_future = modelo.make_future_dataframe(val, periods = horizonte, n_historic_predictions = real_lim)
    forecast = modelo.predict(df_future)
    forecast['Low'] = forecast['yhat1'] - 1.645 * std
    forecast['High'] = forecast['yhat1'] - 0.822 * std
    return True, forecast

def graficar(datos, real_lim, quant, PCH):
    primer_pron = datos.iloc[real_lim-1][['ds','y']]

    datos_reales = datos.iloc[:real_lim].copy()
    datos_reales['y'] = datos_reales['y'].apply(lambda x: max(x, 0))

    datos_pron = datos.iloc[real_lim:].copy()
    datos_pron['yhat1'] = datos_pron['yhat1'].apply(lambda x: max(x, 0))
    datos_pron['Low'] = datos_pron['Low'].apply(lambda x: max(x, 0))
    datos_pron['High'] = datos_pron['High'].apply(lambda x: max(x, 0))

    datos_pron = pd.concat([pd.DataFrame({'ds': [primer_pron['ds']], 'yhat1': [primer_pron['y']],
                                          'Low': [primer_pron['y']], 'High': [primer_pron['y']]}), datos_pron])

    # Graficar los datos reales extendidos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_reales['ds'], y=datos_reales['y'], mode='lines', line=dict(width=3, color='blue'), name='Datos Reales'))

    # Graficar el pronóstico a partir de la posición real_lim
    if PCH != 'FLRD':
      fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['yhat1'].clip(lower=0), mode='lines', line=dict(dash='dot', width=3, color='blue'), name='', showlegend=False))
    else:
      fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['High'].clip(lower=0), mode='lines', line=dict(dash='dot', width=3, color='blue'), name='', showlegend=False))

    fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['Low'].clip(lower=0), mode='lines', fill='tonexty', line=dict(dash='dot', width=3, color='blue'), fillcolor='rgba(173, 216, 230, 0.5)', name='Pronóstico', showlegend=True))

    # Configuración del diseño del gráfico
    fig.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', width=2100, height=450,
                      yaxis=dict(color="black"),
                      font=dict(family="Prompt", color='black'),
                      xaxis=dict(color="black", tickmode='array', tickangle=-90,
                                 tickvals=datos['ds'].dt.strftime('%Y-%m'),
                                 ticktext=datos['ds'].dt.strftime('%Y-%m')),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                  font_size=16, font_color='black'),
                      showlegend=True)

    # Configuración del eje X e Y con el límite inferior en Y establecido en cero
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray', mirror=False,
                     title_text='<b>Tiempo (mes)</b>', titlefont_size=18, tickfont_size=16)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray', mirror=False,
                     title_text='<b>Generación (GW-mes)</b>', titlefont_size=18, tickfont_size=16,
                     tickformat='.2f')  # Límite inferior en Y establecido en 0
    fig.update_traces(hovertemplate='Periodo:</b> %{x}<br><b>Generación:</b> %{y:.2f} GW-mes<extra></extra>')

    return fig


def main():
  font = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;700&display=swap');

        body, h1, h2, h3, h4, h5, h6, p, .stDataFrame, .css-1y0t9o9, .stButton>button, .css-1wa3eu0, .css-10jvk68, .css-1y0t9o9 {
            font-family: 'Prompt', sans-serif !important;
        }

    </style>
  """
  st.set_page_config(page_title="Pronóstico PCH Vatia",page_icon="images/icon.png",layout="wide")
  st.markdown(font, unsafe_allow_html=True)
  est_pron = False
  st.sidebar.image("images/LogoVatia.png",caption="",use_container_width=True)
  st.sidebar.header("Pronósticos Generación PCH")
  st.markdown('<br>', unsafe_allow_html=True)
  st.sidebar.write('**🚨 Notificaciones**')
  st.markdown('<br>', unsafe_allow_html=True)
  with st.expander('📤 **Cargar históricos de generación**',expanded=True):
    PCH_pot_data = carga_archivos(st.file_uploader('','csv'))
    css = '''<style>
      [data-testid='stFileUploader'] {
          width: 100%;
      }
      [data-testid='stFileUploader'] section {
          padding: 0.5em;
          float: left;
      }
      [data-testid='stFileUploader'] section > input + div {
          display: true;
          padding: 0.5em;
      }
      [data-testid='stFileUploader'] section + div {
          float: center;
          padding: 1em;
      }
      [data-testid="stSidebar"] img {
          margin-top: -70px  !important; /* Ajustar según el espacio requerido */
          margin-left: 0px;
      }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    placeholder_1 = st.sidebar.empty()
  if not PCH_pot_data.empty:
    if 'PCH_pot_data_f3' not in st.session_state:
      PCH_pot_data_f2 = obtener_PCH_data(PCH_pot_data)
      st.session_state['PCH_pot_data_f3'] = PCH_preprocess(PCH_pot_data_f2, "2009-10-01")
    PCH_pot_data_f3 = st.session_state['PCH_pot_data_f3']
    PCHS = PCH_pot_data_f3['PLANTA'].unique().tolist()
    descripcion_PCH = {"FLRD": "FLORIDA", "STG1": "SANTIAGO",
                       "MIR1": "MIROLINDO", "INZ1": "INZÁ",
                       "SLV1": "SILVIA", "VNT1": "VENTANA",
                       "RCIO": "RIO RECIO", "OVJ1": "OVEJAS",
                       "SJN1": "SAJANDÍ", "PST1": "PASTALES",
                       "ASN1": "ASNAZÚ" , "LPLO": "RIO PALO",
                       "MND1": "MONDOMO", "CAUC": "Grupo PCH Cauca",
                       "HTOL":"Grupo PCH Hidrotolima", "VAT1": "Total PCHs"}
    PCHS_desc = ['--'] + [f"{pch} - {descripcion_PCH.get(pch,'')}" for pch in PCHS]
    descripcion_to_pch = dict(zip(PCHS_desc[1:], PCHS))
    st.sidebar.divider()
    with st.container(border=True):
      selected_option = st.sidebar.selectbox('**Selecciona una PCH**', PCHS_desc)
      PCH_fil = descripcion_to_pch.get(selected_option, None)
      if PCH_fil == None:
        placeholder_1.warning(":point_down: Por favor selecciona una PCH antes de proceder con su pronóstico.")
      else:
        df_filtrado = PCH_pot_data_f3[PCH_pot_data_f3['PLANTA'] == PCH_fil]
        if PCH_fil not in ['INZ1', 'OVJ1', 'SJN1']:
            df_filtrado = imputar_TS(df_filtrado, 'POT')

        current_date,min_date = df_filtrado['PERIODO'].max(), df_filtrado['PERIODO'].min()
        current_year, current_month = current_date.year, current_date.month-1
        months = ['Ene -', 'Feb -', 'Mar -', 'Abr -', 'May -', 'Jun -',
                  'Jul -', 'Ago -', 'Sep -', 'Oct -', 'Nov -', 'Dic -']

        # Encontrar el valor máximo y el periodo correspondiente
        max_value = round(max(df_filtrado.POT.dropna()), 2)
        max_period = df_filtrado.loc[df_filtrado.POT.idxmax(),'PERIODO'].strftime('%Y - %m')
        expander = st.sidebar.expander('**Información Histórica:**', expanded=True)
        expander.metric(':bulb: Máxima Generación (GW-mes)', max_value,'('+max_period+')', delta_color="off")
        expander.metric(':date: Disponibilidad de datos (AAAA/MM)', current_date.strftime('%Y/%m'))

    with st.expander('📊 **Información de Pronóstico:**',expanded=True):
      col3, col4 = st.columns([1,3])
      if PCH_fil != None:
        col3.metric(':alarm_clock: Periodo Inicial', str(months[current_month+1])+' '+str(current_year))
        horizonte = col4.slider(':calendar: Horizonte de pronóstico (meses)', 1, 15, 15)
        selected_month = months[current_month+1]
        fecha = pd.Timestamp(year=int(current_year), month=months.index(selected_month), day=1)
        #if st.sidebar.button(":chart_with_upwards_trend: :zap: Pronosticar", use_container_width=True):
        placeholder_1.warning("Generando pronóstico para " + PCH_fil + ", Por favor espere.... ⏳")
        modelo, val, quant, dstd = entrenar(df_filtrado, fecha, horizonte)
        est_pron, forecast = pronostico(val, modelo, horizonte, real_lim=15, std=dstd)
        st.plotly_chart(graficar(forecast, 15, quant, PCH_fil), use_container_width=True)

    with st.expander('📥 **Descarga de Resultados:**',expanded=False):
        if est_pron:
          extracto = forecast[['ds', 'Low', 'yhat1']].iloc[15:]
        else:
          extracto = pd.DataFrame(columns=['ds', 'Low', 'yhat1'])

        if PCH_fil != 'FLRD':
          extracto['yhat1'] = extracto['yhat1'].apply(lambda x: max(x, 0))
        else:
          extracto['yhat1'] = forecast['High'].iloc[15:].apply(lambda x: max(x, 0))

        extracto['Low'] = extracto['Low'].apply(lambda x: max(x, 0))
        extracto.columns = ['Periodo', 'Generación Mínima (GW-mes)', 'Generación Máxima (GW-mes)']
        #extracto.columns = ['Periodo', 'Pronóstico (GW-mes)', 'Generación Mínima (GW-mes)', 'Generación Máxima (GW-mes)']
        if extracto.empty:
          pass
        else:
          extracto.set_index('Periodo', inplace=True)
          extracto.index = extracto.index.strftime('%Y-%m')
          extracto = extracto.map(lambda x: f"{x:.3f}")
          placeholder_1.success(":chart_with_upwards_trend: :heavy_check_mark: Proceso Finalizado")
        st.dataframe(extracto.style.map(lambda x: "font-size: 18pt"),#(**{'', }),
                     height=200, width=2000)

  else:
    st.sidebar.warning(":warning: Por favor, carga un archivo para continuar..")

if __name__ == "__main__":
  main()
