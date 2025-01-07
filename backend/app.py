from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# Load and preprocess data
train_data = pd.read_csv('./data/train.csv')
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
categorical_features = train_data.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = train_data.drop(['SalePrice'], axis=1)
X_preprocessed = preprocessor.fit_transform(X)

app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        max_price = float(data['max_price'])
        features = np.array(data['features']).reshape(1, -1)

        # Define the input features
        input_features = {
            'LotArea': features[0][0],
            'BedroomAbvGr': features[0][1],
            'FullBath': features[0][2],
            'YearBuilt': features[0][3],
            'GrLivArea': features[0][4],
            'YearRemodAdd': features[0][5]
        }

        # Create a DataFrame with default values for missing columns
        input_data = pd.DataFrame([input_features])
        for col in train_data.columns:
            if col not in input_data.columns:
                input_data[col] = train_data[col].mode()[0] if train_data[col].dtype == 'object' else 0

        # Ensure the columns match the order in train_data
        input_data = input_data[train_data.columns.drop('SalePrice')]

        # Preprocess the input features
        input_preprocessed = preprocessor.transform(input_data)

        # Compute cosine similarity
        similarities = cosine_similarity(input_preprocessed, X_preprocessed)

        # Filter recommendations based on max_price
        train_data['similarity'] = similarities[0]
        recommendations = train_data[train_data['SalePrice'] <= max_price].sort_values(by='similarity', ascending=False).head(10)

        return jsonify(recommendations[['Id', 'SalePrice', 'similarity']].to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)