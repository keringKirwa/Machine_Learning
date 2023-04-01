import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""" (1) Note that we set the n_init variable to a constant 10 ,  meaning that the algorithm will run 10 times , 
witht different initial centroids, a large value  guarantees good results, but with great computational power (2) Set 
random seed for reproducibility (3) To check the type of an item in Python , then i user the built in type type()  
function. (4)K-means++ is a smart initialization method that tries to place the centroids in a way that speeds up the 
convergence of the algorithm. It does this by selecting the first centroid randomly from the data points, 
and then selecting subsequent centroids from the remaining data points in a way that MAXIMIZES the minimum distance 
between each centroid and the other centroids that have been selected so far. This ensures that the centroids are 
well-spaced and avoids the problem of initializing centroids in close proximity to each other, which can lead to 
suboptimal clustering results.(5) teh number of centroids chosen is the same as the  number of clusters provided."""

np.random.seed(42)

num_customers = 50
num_categories = 3

purchases = np.random.randint(1, 5, size=(num_customers, num_categories))
total_spent = np.random.randint(50, 250, size=num_customers)

# Create DataFrame
data = pd.DataFrame(purchases, columns=['Shoes', 'Clothes', 'Laptops'])
data['Total_Spent(USD)'] = total_spent
data['Number of Purchases'] = data.sum(axis=1)
if __name__ == '__main__':

    X = data[['Shoes', 'Laptops', 'Total_Spent(USD)', 'Clothes', 'Number of Purchases']]
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        print(f"For k={k}, the silhouette score is {silhouette_avg}")

    # Instantiate the KMeans class with the number of clusters we want
    kmeans = KMeans(n_clusters=2, random_state=42)

    # Fit the KMeans model to the data
    kmeans.fit(X)

    """Adding the generated cluster column to the  dataframe.The Kmeans.labels_ contain the  info for the  @customer ,  
    the  cluster label  for each customer, its an array whose length is the same as that of the customers 
    Kmeans will always label the  data it has classified.
    """

    data['Cluster'] = kmeans.labels_

    cluster_stats = data.groupby('Cluster').mean()
    print("Cluster Statistics based on the MEAN :\n", cluster_stats)
    print(data)

    print("All The Customers In Cluster 2(index 1) : \n")
    # same as  cluster2_customers = data[data.Cluster == 1]
    cluster2_customers = data[data["Cluster"] == 1]
    print(cluster2_customers)
