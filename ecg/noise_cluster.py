import pickle

import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model


def label_processing(total_data):

    for label in ['label1', 'label2', 'label3', 'label4', 'label5']:
        total_data.loc[total_data[label] == '8', label] = '2'
        total_data.loc[total_data[label] == '10', label] = '3'
        total_data.loc[total_data[label].isin(['7', '12', '14']), label] = '10'
        total_data.loc[total_data[label].isin(['15', '16', '17', '18', '19']), label] = None

        # label 번호 당기기
        total_data.loc[total_data[label] == '9', label] = '7'
        total_data.loc[total_data[label] == '11', label] = '8'
        total_data.loc[total_data[label] == '13', label] = '9'

    # label processing - if label1 = 1, label2 not None, label1 = label2, label2 = label 3,...
    total_data = total_data.reset_index(drop=True)
    for i in range(len(total_data['label1'])):
        if total_data.loc[i,'label2'] is not None:
            if total_data.loc[i,'label1'] == '1':
                total_data.loc[i,'label1'] = total_data.loc[i,'label2']
                total_data.loc[i,'label2'] = total_data.loc[i,'label3']
                total_data.loc[i,'label3'] = total_data.loc[i,'label4']
                total_data.loc[i,'label4'] = total_data.loc[i,'label5']
                total_data.loc[i,'label5'] = None

    # label processing - if label1 not 1, label 2 not None ,, label1 is None-> remove
    notidx = (((total_data['label1'] != '1') & (total_data['label2'].notnull())) | total_data['label1'].isnull())
    total_data = total_data[~notidx]
    total_data = total_data.reset_index(drop=True)

    return total_data

if __name__ == '__main__':
    my_model = load_model("C:\\Users\\user\\Documents\\SEResNet_6class_NewDBv2_Noise\\result\\"
                          + "SEResNet152_6class_lead2_size2000_ver1_2.h5", custom_objects={'relu6': tf.nn.relu6})
    lead_1_pkl = '/media/sangil/477A156111D616D6/data/gunguk/konkuk_v2_7_1_lead1.pkl'
    load_pickle = open(lead_1_pkl, 'rb')
    load_data = pickle.load(load_pickle)
    load_pickle.close()

    load_data = label_processing(load_data)
    X_l = np.array(load_data.iloc[:, 11:], dtype=float)
    model = Model(inputs=my_model.input, outputs=my_model.get_layer("global_average_pooling2d_51").output)

    get_features = model.predict(X_l)

    num_cluster = 100

    kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_cluster, use_mini_batch=False, distance_metric='cosine')


    def input_fn():
        return tf.compat.v1.train.limit_epochs(
            tf.convert_to_tensor(get_features, dtype=tf.float32), num_epochs=1)


    ####### train #########
    num_iterations = 50

    for _ in range(num_iterations):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()


    # cosine_similarity
    def cos_sim(a, b):
        l2_norm_a = norm(a, axis=1, keepdims=True)
        l2_norm_b = norm(b, axis=1, keepdims=True)
        l2_norm_a = np.divide(a, l2_norm_a, where=l2_norm_a != 0)
        l2_norm_b = np.divide(b, l2_norm_b, where=l2_norm_b != 0)

        return np.dot(l2_norm_a, l2_norm_b.T)


    cosine = cos_sim(get_features, cluster_centers)
    print(cosine.shape)

    cluster_indices = list(kmeans.predict_cluster_index(input_fn))  # find the index of closest clustering center

