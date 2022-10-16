import streamlit as st
from matplotlib import pyplot as plt
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

X_train_feat = pd.read_csv('X_Train_Encoded_FeatEng_data.csv')
X_valid_feat = pd.read_csv('X_Valid_Encoded_FeatEng_data.csv')
y_train = pd.read_csv('y_Train_Encoded_FeatEng_data.csv')
y_valid = pd.read_csv('y_Valid_Encoded_FeatEng_data.csv')

train_feat = X_train_feat.copy()
valid_feat = X_valid_feat.copy()
valid_feat['cluster'] = y_valid
tracks=[train_feat['track_name'],valid_feat['track_name']]
tracks=pd.concat(tracks)
tracks=tracks.drop_duplicates()
df_feat= pd.concat([train_feat, valid_feat], axis=0)

columns = df_feat.columns
cat_index = []
index = 0
for col in columns:
    if df_feat[col].dtype == 'object':
        cat_index.append(index)
    index +=1

class SpotifyRecommender():
    def __init__(self, rec_data):
        #our class should understand which data to work with
        self.rec_data_ = rec_data.copy()
    #if we need to change data
    def change_data(self, rec_data):
        self.rec_data_ = rec_data
    #function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations_knn(self, song_name, amount=1):
        #choosing the data for our song
        try:
            song = self.rec_data_[(self.rec_data_.track_name.str.lower() == song_name.lower())].head(1).values[0] # vector of first time seen song
        except:
            print("This Song Doesn't Exist")
            return
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.track_name.str.lower() != song_name.lower()]
        distances = []
        for r_song in tqdm(res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                #indeces of non-numerical columns
                if not col in [cat_index]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
#         sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        res_data = res_data.drop_duplicates(subset = ['track_name'])
        columns = ['track_name'] # name -> uri
        # mapping
        return res_data[columns][:amount]

    def get_recommendations_cosine(self, song_name, amount=1):
        distances = set()
        #choosing the data for our song
        try:
            index = self.rec_data_[(self.rec_data_.track_name.str.lower() == song_name.lower())].head(1).index.tolist() # vector of first time seen song
            index = index[0]
        except:
            print("This Song Doesn't Exsist")
            return
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.track_name.str.lower() != song_name.lower()]
        return self._get_similar_items_to_user_profile(index, res_data, topn=amount)

def predict(song_name):
    recomender_feat = SpotifyRecommender(df_feat)
    predicts=recomender_feat.get_recommendations_knn(song_name, 10)
    # print(predicts)
    indecies = predicts.index
    song_name_index = np.where(df_feat['track_name'] == song_name)[0][0]
    indecies = indecies.insert(0, song_name_index)
    # print(df_feat.iloc[indecies, :])
    return predicts, df_feat.iloc[indecies, :]
st.set_page_config(
    page_title = 'Song Recommendation',
    page_icon = 'O',
)
def main():
    tracks_ = list(tracks)
    st.title("Spotify Songs Recommendation System")
    st.sidebar.title("Choose Track")
    st.markdown(" Welcome To Our Recommendation System")
    st.warning('Only Enter Track from the list provided')
    select = st.sidebar.selectbox('Choose Track', tracks_, key='1')
    submit = st.button('Recommend me 10 similar songs')
    if submit:
        if select:
            with st.spinner('Predicting...'):
                # time.sleep(2)
                song_name= select
                prediction,locs=predict(song_name)
                st.table(prediction)
        else:
            st.error('Please Enter All the Details')
if __name__ == '__main__':
    main()
