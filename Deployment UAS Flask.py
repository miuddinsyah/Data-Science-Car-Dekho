from flask import Flask, jsonify, request, render_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.svm import SVR
import random

app = Flask(__name__, template_folder='templates')

# Load and preprocess dataset
data = pd.read_csv('Dataset Viechle/Car details v3.csv')
data = data.dropna()
data = data.drop_duplicates(subset=['name', 'year', 'selling_price', 'km_driven', 'seller_type', 'mileage'], keep='first')

# Extract numerical values
data['mileage'] = data['mileage'].str.extract(r'(\d+\.\d+)').astype(float)
data['engine'] = data['engine'].str.extract(r'(\d+)').astype(float)
data['max_power'] = data['max_power'].str.extract(r'(\d+\.\d+)').astype(float)

data = data.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtypes == 'object' else col)
data = data.dropna()

X = data[['year', 'km_driven', 'mileage', 'engine', 'max_power']]
y_reg = data['selling_price']
y_clf = (y_reg > y_reg.median()).astype(int)

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Models
linear_model = LinearRegression()
linear_model.fit(X_train, y_reg_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    if request.method == 'POST':
        try:
            year = float(request.form['year'])
            km_driven = float(request.form['km_driven'])
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])

            input_data = pd.DataFrame([[year, km_driven, mileage, engine, max_power]],
                                      columns=['year', 'km_driven', 'mileage', 'engine', 'max_power'])
            prediction = linear_model.predict(input_data)[0]

            y_pred = linear_model.predict(X_test)
            mse = mean_squared_error(y_reg_test, y_pred)

            return render_template('linear_regression.html', prediction=prediction, mse=mse)
        except Exception as e:
            return render_template('linear_regression.html', error=str(e))

    random_data = {
        'year': random.randint(2000, 2022),
        'km_driven': random.randint(10000, 200000),
        'mileage': round(random.uniform(10, 25), 1),
        'engine': random.randint(1000, 3000),
        'max_power': random.randint(50, 200),
    }
    return render_template('linear_regression.html', random_data=random_data)

@app.route('/svr', methods=['GET', 'POST'])
def svr_regression():
    if request.method == 'POST':
        try:
            c = float(request.form['c'])
            kernel = request.form['kernel']
            gamma = request.form['gamma']

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            svr_model = SVR(C=c, kernel=kernel, gamma=gamma)
            svr_model.fit(X_train_scaled, y_reg_train)

            y_pred = svr_model.predict(X_test_scaled)
            mse = mean_squared_error(y_reg_test, y_pred)

            return render_template('svr.html', mse=mse)
        except Exception as e:
            return render_template('svr.html', error=str(e))

    return render_template('svr.html')

@app.route('/knn', methods=['GET', 'POST'])
def knn_classification():
    if request.method == 'POST':
        try:
            neighbors = int(request.form['neighbors'])
            weights = request.form['weights']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

            knn_model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
            knn_model.fit(X_train_clf, y_train_clf)

            y_pred_clf = knn_model.predict(X_test_clf)
            report = classification_report(y_test_clf, y_pred_clf)

            return render_template('knn.html', report=report)
        except Exception as e:
            return render_template('knn.html', error=str(e))

    return render_template('knn.html')

@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    if request.method == 'POST':
        try:
            max_depth = int(request.form['max_depth'])
            min_samples_split = int(request.form['min_samples_split'])

            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

            tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            tree_model.fit(X_train_clf, y_train_clf)

            y_pred_clf = tree_model.predict(X_test_clf)
            report = classification_report(y_test_clf, y_pred_clf)

            fig, ax = plt.subplots(figsize=(70, 20))
            plot_tree(tree_model, feature_names=X.columns, class_names=['Low', 'High'], filled=True, ax=ax)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            tree_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            return render_template('decision_tree.html', report=report, tree_image=tree_image)
        except Exception as e:
            return render_template('decision_tree.html', error=str(e))

    return render_template('decision_tree.html')

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_clustering():
    if request.method == 'POST':
        try:
            n_clusters = int(request.form['n_clusters'])

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
            ax.set_title(f"K-Means Clustering with {n_clusters} Clusters")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            plt.colorbar(scatter, ax=ax)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            cluster_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            result = f"K-Means Clustering Completed with {n_clusters} Clusters."
            return render_template('kmeans.html', result=result, cluster_image=cluster_image)
        except Exception as e:
            return render_template('kmeans.html', error=str(e))

    return render_template('kmeans.html')
@app.route('/eda')
def eda():
    """Render the EDA menu."""
    return render_template('eda.html')

@app.route('/eda/show_dataset')
def show_dataset():
    """Display the first 20 rows of the dataset."""
    dataset_html = data.head(20).to_html(classes='table table-striped', index=False)
    return render_template('show_dataset.html', table=dataset_html)

@app.route('/eda/show_heatmap')
def show_heatmap():
    """Generate a heatmap of correlations."""
    try:
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        plt.close()
        return render_template('show_heatmap.html', image=image_base64)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/eda/correlation')
def correlation():
    """Generate a pairplot for selected features."""
    try:
        selected_features = ['owner', 'km_driven', 'selling_price', 'fuel']
        sns.pairplot(data[selected_features], hue='fuel', diag_kind='kde', markers=["o", "s", "D", "^"])

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        plt.close()
        return render_template('show_correlation.html', image=image_base64)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/eda/selling_price_count')
def selling_price_count():
    """Generate a histogram for the selling price."""
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='selling_price', bins=100, kde=True, alpha=0.4, line_kws={'linestyle': '--', 'lw': 2.5})
        plt.title("Selling Price Distribution")
        plt.xlabel("Selling Price")
        plt.ylabel("Count")

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        plt.close()
        return render_template('show_histogram.html', image=image_base64)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
