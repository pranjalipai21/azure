from flask import Flask,request, make_response, flash, render_template
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

app = Flask(__name__)
games = pandas.read_csv("games.csv")

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/input', methods=['POST'])
def inpt():
    num = int(float(request.form['num']))
    games = pandas.read_csv("games.csv")
    kmeans_model = KMeans(n_clusters=num, random_state=1)
    # Get only the numeric columns from games.
    good_columns = games._get_numeric_data()
    # Fit the model using the good columns.
    kmeans_model.fit(good_columns)
    # Get the cluster assignments.
    labels = kmeans_model.labels_
    pca_2 = PCA(2)
    # Fit the PCA model on the numeric columns from earlier.
    plot_columns = pca_2.fit_transform(good_columns)
    # Make a scatter plot of each game, shaded according to cluster assignment.
    plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
    # Show the plot.
    #plt.show()
    plt.savefig('static/data.png')
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host="0.0.0.0",debug ='True')