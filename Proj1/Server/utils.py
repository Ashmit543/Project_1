import json
import pickle
import numpy as np

__locations = None
__data_column = None
__model = None


def get_location_names():
    return __locations

def get_estimated_price(locations,sqft,bhk,bath):
    load_saved_artifacts()  # Making sure to load the artifacts before using them
    try:
        loc_index = __data_column.index(locations.lower())
    except:
        loc_index = -1

    p = np.zeros(len(__data_column))
    p[0] = sqft
    p[1] = bath
    p[2] = bhk
    if loc_index >= 0:
        p[loc_index] = 1
    return round(__model.predict([p])[0], 2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_column
    global __locations

    file_path = "D:/DataAnalytics/BHP/Server/artifacts/columns.json"
    with open(file_path, "r") as f:
        __data_column = json.load(f)['data_columns']
        __locations = __data_column[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        file_path = "D:/DataAnalytics/BHP/Server/artifacts/banglore_home_prices_model.pickle"
        with open(file_path, 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) 
    print(get_estimated_price('5th block hbr layout', 1000, 3, 3))

