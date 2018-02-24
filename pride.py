# import os
# import pprint
# import logging
# from flask import Flask

# app = Flask(__name__)

# logging.basicConfig(level=logging.DEBUG)

# @app.route('/')
# def hello():
#     return 'Hello World!\n'

# port = os.getenv('VCAP_APP_PORT', '5000')
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=int(port))

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn import linear_model
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/currentScore', methods=['GET'])
def getScore():
    # return jsonify({'score': 1})


    with open('PrideAtWork.csv', 'r') as f:
        df = pd.read_csv(f, sep=',', low_memory=False, names=['BenefitsAndPolicies','Training','DataCollectionAndAnalysis','NetworkandMentorship','DevelopmentAndEngagement','Support','Year','Name','Score'], skiprows=1)

    new_df_without_name= df.drop('Name', axis=1)
    new_df= new_df_without_name.drop('Year', axis=1)

    X, y = new_df.iloc[:,:-1], new_df.iloc[:, -1]

    reg=linear_model.LinearRegression()
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    reg.fit(x_train, y_train)
    a =reg.score(x_test, y_test)
    return jsonify({'score':a})


if __name__ == '__main__':
    app.run(debug=True)