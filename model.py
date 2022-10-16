import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression #mutal importance of feature
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity # Content Filtering
from tqdm import tqdm as tqdm

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# X_train = pd.read_csv('X_Train_Encoded_FeatEng_data_PCA.csv')
# X_valid = pd.read_csv('X_Valid_Encoded_FeatEng_data_PCA.csv')
X_train_feat = pd.read_csv('X_Train_Encoded_FeatEng_data.csv')
X_valid_feat = pd.read_csv('X_Valid_Encoded_FeatEng_data.csv')
y_train = pd.read_csv('y_Train_Encoded_FeatEng_data.csv')
y_valid = pd.read_csv('y_Valid_Encoded_FeatEng_data.csv')

# mi_scores = make_mi_scores(X_train.iloc[:, :-1], y_train['Cluster'], discrete_features=False)
# train = X_train.copy()
train_feat = X_train_feat.copy()
# valid = X_valid.copy()
valid_feat = X_valid_feat.copy()
# valid['cluster'] = y_valid
valid_feat['cluster'] = y_valid
tracks=[train_feat['track_name'],valid_feat['track_name']]
tracks=pd.concat(tracks)
tracks=tracks.drop_duplicates()
# df = pd.concat([train, valid], axis=0)
df_feat = pd.concat([train_feat, valid_feat], axis=0)

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
    def _get_similar_items_to_user_profile(self, song_id, res_data, topn=1000):
        res_data = res_data.drop('track_name', axis=1).copy()
        #Computes the cosine similarity between the user profile and all item profiles
        series_1 = self.rec_data_.iloc[song_id].copy()
        series_1 = series_1.drop(labels = ['track_name'])
        cosine_similarities = cosine_similarity(np.array(series_1).reshape(1, -1), res_data)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(i, self.rec_data_.iloc[i, cat_index[0]], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[2])
        similar_items_df = pd.DataFrame(similar_items, columns=['index', 'track_name', 'Similarity'])
        similar_items_df = similar_items_df.drop_duplicates(subset = ['track_name'])
        return similar_items_df
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
