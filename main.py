import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import model

#reading Ekg Data to be plotted
# def load_data(select):
#     if select == "EMG Sample Signal":
#         column_names = ['emg', 't']
#         mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
#     elif select == 'EKG Sample Signal':
#         column_names = ['ekg', 't']
#         mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
#     elif select =="ECG Sample Signal":
#         data = np.loadtxt('../Interactive-Dashboards-With-Streamlit/ECG.dat',unpack=True)
#         mvc1 = pd.DataFrame(data)
#         mvc1.columns=['ECG']
#     else:
#         column_names = ['eeg', 't']
#         mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
#     return mvc1


def main():
    tracks = list(model.tracks)
    st.title("Spotify Songs Recommendation System")
    st.sidebar.title("Choose Track")
    st.markdown(" Welcome To Our Recommendation System")
    st.warning('Only Enter Track from the list provided')
    select = st.sidebar.selectbox('Choose Track', tracks, key='1')
    submit = st.button('Recommend me 10 similar songs')
    if submit:
        if select:
            with st.spinner('Predicting...'):
                # time.sleep(2)
                song_name= select
                prediction,locs=model.predict(song_name)
                st.table(prediction,locs)
                # st.info(f" {prediction} ")
        else:
            st.error('Please Enter All the Details')
if __name__ == '__main__':
    main()