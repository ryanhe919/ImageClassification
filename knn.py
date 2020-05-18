from sklearn.neighbors import KNeighborsClassifier
import joblib
from ImageClassification import data_process
import os

MODEL_DIR = './Model/'


# training through knn model.
# input: train_data, train_labels
# output: none
# It will generate a knn model store in Model directory.
def training_knn_model(train_data, train_labels):
    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(train_data.reshape(-1, 3072), train_labels)
    joblib.dump(neigh, os.path.join(MODEL_DIR, 'knn.model'))


# predict through knn model.
# input: test_data
# output: predict_result
def predict_by_knn(test_data):
    neigh = joblib.load(os.path.join(MODEL_DIR, 'knn.model'))
    result = neigh.predict(test_data.reshape(-1, 3072))
    return result


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data_process.read_cifar_10()
    # training_knn_model(train_data, train_labels)
    print(predict_by_knn(test_data[0:100]), test_labels[0:100])
