import numpy as np
import pandas as pd
from scipy import linalg
#from sklearn.datasets import make_sparse_spd_matrix
from sklearn.impute import SimpleImputer
import networkx as nx
from sknetwork.clustering import Louvain
from sknetwork.visualization import svg_graph, svg_bigraph

# import data and standardize
dat = pd.read_csv("hapo_metabolomics_2020.csv")
#do simple imputation of mean of numberic columns 
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
num_col = dat.select_dtypes(include = 'number').columns
newdat = dat.drop(columns = (['id', 'anc_gp', 'fpg']))
#newdat[num_col] = pd.DataFrame(imp.fit_transform(dat[num_col]), columns = num_col)
newdat = pd.DataFrame(imp.fit_transform(newdat))
normalized_df=(newdat-newdat.mean())/newdat.std()
X = normalized_df

# estimate the covariance 
n_samples = len(dat)

from sklearn.covariance import GraphicalLassoCV, ledoit_wolf

emp_cov = np.dot(X.T, X) / n_samples

model = GraphicalLassoCV()
cov = model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_

lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# plot the results

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [
    ("Empirical", emp_cov),
    ("Ledoit-Wolf", lw_cov_),
    ("GraphicalLassoCV", cov_)#,
    #("True", cov),
]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(
        this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
    )
    plt.xticks(())
    plt.yticks(())
    plt.title("%s covariance" % name)


# plot the precisions
precs = [
    ("Empirical", linalg.inv(emp_cov)),
    ("Ledoit-Wolf", lw_prec_),
    ("GraphicalLasso", prec_)#,
    #("True", prec),
]
vmax = 0.9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.imshow(
        np.ma.masked_equal(this_prec, 0),
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap=plt.cm.RdBu_r,
    )
    plt.xticks(())
    plt.yticks(())
    plt.title("%s precision" % name)
    if hasattr(ax, "set_facecolor"):
        ax.set_facecolor(".7")
    else:
        ax.set_axis_bgcolor(".7")
        

# plot the model selection metric
#plt.figure(figsize=(4, 3))
#plt.axes([0.2, 0.15, 0.75, 0.7])
plt.plot(model.cv_results_["alphas"], model.cv_results_["mean_test_score"], "o-")
plt.axvline(model.alpha_, color=".5")
plt.title("Model selection")
plt.ylabel("Cross-validation score")
plt.xlabel("alpha")

plt.show()

# now get adjacency matrix from graphical lasso results

G = nx.Graph(model.get_precision())
adjacency = nx.to_scipy_sparse_matrix(G)
algo = Louvain()
algo.fit(adjacency) 
adjacency = G._adj
test = nx.to_scipy_sparse_matrix(adjacency)


embedding = G.fit_transform
position = G.position
labels = graph.labels

# test from doc
from IPython.display import SVG
from sknetwork.data import karate_club
adjacency = karate_club()
algo = Louvain()
algo.fit(adjacency)
labels = algo.labels_
graph = karate_club(metadata=True)
adjacency = graph.adjacency
position = graph.position
labels = graph.labels
# graph
image = svg_graph(adjacency, position, labels=labels)
SVG(image)