# Core Pkg
import streamlit as st
import os
import json
import requests

# EDA Pkgs
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg') 
import joblib
API_KEY = '5b931a847b5cc20c2af0daa22f860669ad98da028c9123c76e2bb3a2'
API_URL = 'https://api.ipdata.co/'

col_names = ['DE_wind_generation_actual','DE_solar_generation_actual','cumulated hours', 'lat', 'lon', 'v1', 'v2', 'v_50m', 'h1', 'h2', 'z0',
       'SWTDN', 'SWGDN', 'T', 'rho', 'p']

def load_data(dataset):
	df = pd.read_csv(dataset,names=col_names)
	return df


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

data2 = load_data('testingf.csv')
data2=data2.drop(data2.index[0])


aa=[]
aa=data2['v1'].unique()
v1h1_label = aa
ab=[]
ab=data2['v2'].unique()
v2h2_label = ab
ac=[]
ac=data2['v_50m'].unique()
v50m_label = ac
ad=[]
ad=data2['z0'].unique()
z0_label=ad
ae=[]
ae=data2['SWTDN'].unique()
swtdn_label=ae
af=[]
af=data2['SWGDN'].unique()
swgdn_label=af
ag=[]
ag=data2['T'].unique()
t_label=ag



def main():
	"""Car Evaluation with ML Streamlit App"""
	
	st.title("Turbine energy prediction using weather conditions(click '>' to know prediction)")
	st.subheader(" ML App")
	#st.image(load_image("car_images/car.jpg"),width=300, caption='Images')

	activities = ['EDA','Prediction','About']
	choices = st.sidebar.selectbox("Select Activity",activities)

	if choices == 'EDA':
	 st.subheader("EDA")
	 st.title('Our Dataset')
	 st.dataframe(data2)
	 def query_ip(ips):
	  df2 = pd.DataFrame()
	  for ip in ips.split(','):
	   ip = ip.strip()
	   response = requests.get(f'{API_URL}{ip}?api-key={API_KEY}')
	   if response.status_code == 200:
	    data = json.loads(response.content)
	    df2 = df2.append(pd.DataFrame(pd.json_normalize(data)))
            
	  return df2


	 st.title('IP Geolocation')
	 persons = st.text_input('enter a 4-digit ip (ex-8.8.8.8:United states etc) of country to show your country details you can enter multiple ip separated by comma')
	 ips = st.text_area('IP addresses of country separated by comma(EX-8.8.8.8,4.4.4.4)', persons)
	 if st.button('Click to Get location'):
	  df1 = query_ip(ips)
	  st.dataframe(df1)
	  st.table(df1[['ip', 'emoji_flag', 'country_name', 'city', 'asn.name', 'asn.type']].reset_index())
	  st.deck_gl_chart(
	 layers=[
      {
        'data': df1,
        'type': 'TextLayer',
        'getText': 'ip',
      },
      {
        'data': df1,
         'type': 'ScatterplotLayer',
         'getRadius': 1e2,
      }
	 ]
	 )








	 st.title('Overview of dataset')
	 data = load_data('testingf.csv')
	 data=data.drop(data.index[0])
	 st.dataframe(data.head(5))
	 
	 data4=data.drop(data.index[:-4])
	 if st.checkbox("Show Summary of Dataset"):
	  st.write(data.describe())

		# Show Plots
	 if st.checkbox("Simple Value Plots "):
	 	st.write(data4['DE_wind_generation_actual'].value_counts().plot(kind='bar'))
	 	st.pyplot()


		
	 if st.checkbox("Select Columns To Show"):
	  all_columns = data.columns.tolist()
	  selected_columns = st.multiselect('Select',all_columns)
	  new_df = data[selected_columns]
	  st.dataframe(new_df)

	 if st.checkbox("Pie Plot"):
	  st.write(data4['DE_wind_generation_actual'].value_counts().plot.pie(autopct="%1.1f%%"))
	  st.pyplot()			


	if choices == 'Prediction':
	 st.subheader("Prediction")

	 b = st.selectbox('velocity at height 2m',tuple(v1h1_label))
	 c = st.selectbox('velocity at height 10m',tuple(v2h2_label))
	 d = st.selectbox('velocity at height 50m',tuple(v50m_label))
	 e = st.selectbox("roughness length",tuple(z0_label))
	 f = st.selectbox('atmosphere horizontal radiation',tuple(swtdn_label))
	 g = st.selectbox("ground horizontal radiation ",tuple(swgdn_label))
	 h = st.selectbox("temperature",tuple(t_label))

		

		
	 pretty_data = {
	  "velocity(2m)":b,
	  "velocity(10m)":c,
	  "velocity(50m)":d,
	  "roughness length":e,
	  "atmosphere horizontal radiation":f,
	  "ground horizontal radiation":g,
	  "temperature":h
		}
	 st.subheader("Options Selected")
	 st.json(pretty_data)

	 st.subheader("weather data selected as for wind energy prediction")
		# Data To Be Used
	 sample_data1 = [b,c,d,e]
	 st.write(sample_data1)
	 st.subheader("weather data selected as for solar energy prediction")
	 sample_data2 = [f,g,h]
	 st.write(sample_data2)
	 prep_data1 = np.array(sample_data1,dtype='float64').reshape(1, -1)
	 prep_data2 = np.array(sample_data2,dtype='float64').reshape(1, -1)
	 model_choice = st.selectbox("Model Type(prediction for wind energy OR solar energy)",['wind energy prediction','solar energy prediction'])
	 if st.button('Evaluate'):
	  if model_choice == 'wind energy prediction':
	   predictor = load_prediction_models("logittony_w_model.pkl")
	   prediction = predictor.predict(prep_data1)
	   st.write(prediction)
	   st.write("the wind energy prediction based on weather condition is")
	  
	  if model_choice == 'solar energy prediction':
	   predictor = load_prediction_models("logitstark_w_model.pkl")
	   prediction = predictor.predict(prep_data2)
	   st.write(prediction)
	   st.write("the solar energy prediction based on weather condition is")
	  
	  if model_choice == 'MLP classifier':
	   predictor = load_prediction_models("nn_clf_car_model.pkl")
	   prediction = predictor.predict(prep_data)
	   st.write(prediction)


	  st.success(prediction)
	if choices == 'About':
	 st.subheader("About")
	 st.write("This web app gives the prediction on energy capacity on based on weather condition")


if __name__ == '__main__':
	main()
	
