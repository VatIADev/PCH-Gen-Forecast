import streamlit as st
import pandas as pd
import numpy as np
import pandasql as psql
import torch.nn as nn
import plotly.graph_objects as go
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
set_random_seed(42)

@st.cache_data
def carga_archivos(archivo):
  DB = ""
  if archivo is not None:
    DB = pd.read_csv(archivo).drop(columns=['TIPO', 'VERSION'])
    print("Base de datos cargada")
    st.sidebar.success("Base de datos cargada")
  else:
    return pd.DataFrame()
  return DB

#WHERE fecha BETWEEN '2004-01-01' AND '2024-09-01'
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
    umbral = df[columna_valores].max() * (15 / 100)
    df[columna_valores] = df[columna_valores].apply(lambda x: x if x >= umbral else float('nan'))
    df[columna_valores] = df[columna_valores].interpolate(method='cubic')
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
  return PCH_data

def setpoint(planta, horizonte):
    base_params = {
        'n_changepoints': 10, 'changepoints_range': 0.9, 'growth': 'discontinuous', 'optimizer': 'AdamW',
        'epochs': 300, 'n_forecasts': horizonte, 'quantiles': []#[0.1, 0.9]
    }
    specific_params = {
        'OVJ1': {},
        'FLRD': {'n_changepoints': 10, 'changepoints_range': 0.8, 'growth': 'linear'},
        'MIR1': {'n_changepoints': 6, 'growth': 'linear', 'changepoints_range': 0.8},
        'INZ1': {},
        'RCIO': {'n_changepoints': 15, 'changepoints_range': 0.8, 'loss_func':nn.HuberLoss},
        'PST1': {'n_changepoints': 36, 'loss_func':nn.HuberLoss},
        'VNT1': {'n_changepoints': 10, 'changepoints_range': 0.8, 'growth': 'linear'},
        'STG1': {'n_changepoints': 6, 'changepoints_range': 0.9, 'loss_func':nn.HuberLoss, 'growth': 'linear'},
        'SJN1': {'n_changepoints': 10, 'changepoints_range': 0.8, 'loss_func':nn.HuberLoss},
        'ASN1': {'n_changepoints': 5, 'changepoints_range': 0.9, 'loss_func':nn.HuberLoss},
        'LPLO': {'n_changepoints': 10, 'loss_func':nn.HuberLoss, 'changepoints_range': 0.75, 'growth': 'linear'},
        'MND1': {'n_changepoints': 3, 'changepoints_range': 0.9, 'loss_func':nn.HuberLoss},
        'SLV1': {'n_changepoints': 13, 'changepoints_range': 0.9, 'loss_func':nn.HuberLoss, 'growth': 'linear'}
    }
    params = {**base_params, **specific_params.get(planta, {})}

    return params

def entrenar(datos,fecha,horizonte):
  params = setpoint(datos['PLANTA'].unique()[0],horizonte)
  #st.write(params)
  datos.drop(["PLANTA"], axis=1, inplace=True)
  datos.columns = ['ds', 'y']
  train,test = datos[datos.ds <= fecha], datos[datos.ds > fecha]
  m = NeuralProphet(**params)
  m.add_country_holidays("CO")
  train, val = m.split_df(datos, freq='M', valid_p=0.2)
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
  forecast['High'] = forecast['yhat1'] + 1.645 * std
  return forecast

def graficar(datos, real_lim, quant):
    # Tomar el primer punto de los datos de pronóstico
    primer_pron = datos.iloc[real_lim]

    # Añadir el primer punto del pronóstico al final de los datos reales para unificar la serie
    datos_reales = datos.iloc[:real_lim].copy()
    datos_reales = pd.concat([datos_reales, pd.DataFrame({'ds': [primer_pron['ds']], 'y': [primer_pron['yhat1']]})])

    # Convertir valores negativos en cero en los datos reales y pronóstico
    datos_reales['y'] = datos_reales['y'].apply(lambda x: max(x, 0))

    datos_pron = datos.iloc[real_lim:].copy()
    datos_pron['yhat1'] = datos_pron['yhat1'].apply(lambda x: max(x, 0))
    datos_pron['Low'] = datos_pron['Low'].apply(lambda x: max(x, 0))
    datos_pron['High'] = datos_pron['High'].apply(lambda x: max(x, 0))

    # Graficar los datos reales extendidos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_reales['ds'], y=datos_reales['y'], mode='lines', name='Datos Reales'))

    # Graficar el pronóstico a partir de la posición real_lim
    fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['yhat1'].clip(lower=0), mode='lines', name='Pronóstico'))
    fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['Low'].clip(lower=0), mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=datos_pron['ds'], y=datos_pron['High'], mode='lines', fill='tonexty', line=dict(width=0), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=True, name='Intervalo'))

    # Configuración del diseño del gráfico
    fig.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', width=2000, height=450,
                      yaxis=dict(color="black"),
                      font=dict(family="Futura LT Medium", color='black'),
                      xaxis=dict(color="black", tickmode='array', tickangle=-30,
                                 tickvals=datos['ds'].dt.to_period('M').astype(str),
                                 ticktext=datos['ds'].dt.to_period('M').astype(str)),
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
  st.set_page_config(page_title="Pronóstico PCH Vatia",page_icon="/content/icon.png",layout="wide")
  st.title("Pronóstico PCH V1.0")
  st.sidebar.title("Históricos de PCH")
  PCH_pot_data = carga_archivos(st.sidebar.file_uploader('Cargar historicos de Generación','csv'))
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
                       "MND1": "MONDOMO"}
    PCHS_desc = ['--'] + [f"{pch} - {descripcion_PCH.get(pch,'')}" for pch in PCHS]
    descripcion_to_pch = dict(zip(PCHS_desc[1:], PCHS))
    selected_option = st.selectbox('Selecciona una PCH', PCHS_desc)
    PCH_fil = descripcion_to_pch.get(selected_option, None)
    if PCH_fil == None:
      st.warning("Por favor selecciona una PCH antes de proceder con su pronóstico.")
    else:
      df_filtrado = PCH_pot_data_f3[PCH_pot_data_f3['PLANTA'] == PCH_fil]
      df_filtrado = imputar_TS(df_filtrado, 'POT')
      print(df_filtrado)
      horizonte = st.sidebar.slider('Horizonte de pronóstico (Meses)', 1, 15, 15)
      current_date,min_date = df_filtrado['PERIODO'].max(), df_filtrado['PERIODO'].min()
      current_year, current_month = current_date.year, current_date.month-1
      years = list(range(current_year - 5, current_year + 1))
      months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',
                'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

      col1, col2 = st.columns(2)
      # Encontrar el valor máximo y el periodo correspondiente
      max_value = round(max(df_filtrado.POT.dropna()), 3)
      max_period = df_filtrado.loc[df_filtrado.POT.idxmax(),'PERIODO'].strftime('%Y - %m')
      # Mostrar el metric con el valor máximo y el periodo
      col2.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">
            Máxima Generación (GW-mes)
        </div>
        <div style="font-size:32px;">
            {max_value} <span style="font-size:16px; color:gray;">({max_period})</span>
        </div>""",
        unsafe_allow_html=True)
      col1.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">
            Disponibilidad Históricos (AAAA/MM)
        </div>
        <div style="font-size:32px;">
            {current_date.strftime('%Y/%m')}
        </div>""",
        unsafe_allow_html=True)

      with col1:
        selected_year = st.selectbox('Año', years, index=years.index(current_year))
      with col2:
        selected_month = st.selectbox('Mes', months, index=current_month+1)

      fecha = pd.Timestamp(year=int(selected_year), month=months.index(selected_month)+1, day=1)
      placeholder = st.empty()
      if st.sidebar.button("Generar pronóstico", use_container_width=True):
        placeholder.warning("Generando pronóstico para " + PCH_fil + ", Por favor espere.... ⏳")
        modelo, val, quant, dstd = entrenar(df_filtrado, fecha, horizonte)
        forecast = pronostico(val, modelo, horizonte, real_lim=15, std=dstd)
        extracto = forecast[['ds', 'yhat1', 'Low', 'High']].iloc[15:]
        extracto['yhat1'] = extracto['yhat1'].apply(lambda x: max(x, 0))
        extracto['Low'] = extracto['Low'].apply(lambda x: max(x, 0))
        extracto['High'] = extracto['High'].apply(lambda x: max(x, 0))
        extracto.columns = ['Periodo', 'Pronóstico (GW-mes)', 'Generación Mínima (GW-mes)', 'Generación Máxima (GW-mes)']
        extracto.set_index('Periodo', inplace=True)
        extracto.index = extracto.index.strftime('%Y-%m')

        # Formatear todas las columnas a dos decimales
        extracto = extracto.map(lambda x: f"{x:.2f}")

        placeholder.success("Proceso Finalizado.")
        st.plotly_chart(graficar(forecast, 15, quant), use_container_width=True)
        st.dataframe(extracto, height=200, width=2000)
  else:
    st.warning("Por favor carga un archivo para continuar..")

if __name__ == "__main__":
  main()