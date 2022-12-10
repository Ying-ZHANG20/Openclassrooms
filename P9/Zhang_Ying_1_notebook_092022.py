#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Toutes données et analyses sont focalisées sur l’année 2017

import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt
from functions import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
import seaborn as sns



# # I. Data Preparation

# ## 1. pop

# In[2]:


# Import des donnees de population 

pop = pd.read_csv('Population.csv')


# In[3]:


pop.head()


# In[4]:


pop['Valeur']= pop['Valeur'] * 1000 # Change de l'unite de population a 1 personne


# In[5]:


pop = pop[['Zone','Année','Valeur']]


# In[6]:


pop = pop.loc[pop['Année'] == 2017] # Population par pays en 2017


# In[7]:


pop = pop[['Zone','Valeur']] # éliminer les colonnes que nous n'utiliserons pas
pop = pop.rename(columns={'Valeur': 'Population'})
pop.reset_index(drop=True, inplace=True )


# In[8]:


pop['Zone'].value_counts() # 236 unique valeurs pour Zone


# In[9]:


pop.info() # 236 unique non-null valeurs pour les pays


# In[10]:


pop.head()


# ## 2. pib

# In[11]:


# Import des donnees de PIB 2017 Valeur US $ par habitant

pib = pd.read_csv('PIB.csv')


# In[12]:


pib.head()


# In[13]:


pib = pib[['Zone','Valeur']] # éliminer les colonnes que nous n'utiliserons pas
pib = pib.rename(columns={'Valeur': 'PIB par habitant'})
pib.head()


# In[14]:


pib.info() # 211 unique non-null valeurs pour les pays


# ## 3. politic

# In[15]:


# Import des donnees de stabilite politique en 2017

politic = pd.read_csv('Politic.csv')
politic = politic.rename(columns={'Fr': 'Zone'})


# In[16]:


politic.head()


# In[17]:


politic.info() # 198 unique non-null valeurs pour les pays


# ## 4. dispogl

# In[18]:


# Import des donnees de la disponibilité alimentaire (Kcal/personne/jour) en 2017
dispogl = pd.read_csv('Kcal.csv')


# In[19]:


dispogl.head()


# In[20]:


dispogl = dispogl[['Zone','Valeur']] # éliminer les colonnes que nous n'utiliserons pas
dispogl = dispogl.rename(columns={'Valeur': 'Kcal/personne/jour'})
dispogl.head()


# In[21]:


dispogl.info() # 180 unique non-null valeurs pour les pays


# ## 5. protein

# In[22]:


# Import des donnees de la disponibilité totale en protéines et la disponibilité en protéines uniquement d’origine animales, en g/hab/j en 2017
protein = pd.read_csv('Protein.csv')


# In[23]:


protein.head()


# In[24]:


protein = protein[['Zone','Produit','Valeur']] # éliminer les colonnes que nous n'utiliserons pas


# In[25]:


protein.head()


# In[26]:


protein = protein.pivot_table('Valeur', ['Zone'], 'Produit') # éliminer les colonnes que nous n'utiliserons pas
protein.reset_index(drop=False, inplace=True )
protein = protein.rename(columns={'Produits Animaux': 'disponibilité en protéines origine animales g/hab/j', 'Total General': 'disponibilité totale en protéines g/hab/j'})


# In[27]:


protein.head()


# In[28]:


protein.info() # 180 unique non-null valeurs pour les pays


# ## 6. prix

# In[29]:


# Import des donnees de Prix à la Production de volaille (indice) en 2017
prix = pd.read_csv('Prix.csv')


# In[30]:


prix.head()


# In[31]:


prix = prix[['Zone','Valeur']] # éliminer les colonnes que nous n'utiliserons pas
prix = prix.rename(columns={'Valeur': 'Prix Production Volaille (Indice)'})
prix.head()


# In[32]:


prix.info() # 142 unique non-null valeurs pour les pays


# ## 7. volaille

# In[33]:


# Import des donnees spécifiques à la viande de volailles (Milliers de tonnes) en 2017
volaille = pd.read_csv('Volaille.csv')
volaille.head()


# In[34]:


volaille = volaille[['Zone','Élément','Valeur']] # éliminer les colonnes que nous n'utiliserons pas
volaille.head()


# In[35]:


volaille = volaille.pivot_table('Valeur', ['Zone'], 'Élément')
volaille.reset_index(drop=False, inplace=True )
volaille.head()


# In[36]:


volaille.info()  # 178 unique non-null valeurs pour les pays


# ## 8. jointure de tous les dataframes

# In[37]:


# jointure de tous les dataframes precedents pour contruire notre data
data = pop.merge(pib,on ='Zone', how='left').merge(dispogl,on ='Zone', how='left').merge(protein,on ='Zone', how='left').merge(volaille,on ='Zone', how='left').merge(prix,on ='Zone', how='left').merge(politic,on ='Zone', how='left')


# In[38]:


data.head()


# In[39]:


data.drop(columns= ['Country', 'Year'],inplace=True) # éliminer les colonnes que nous n'utiliserons pas


# In[40]:


data.head()


# In[41]:


data.info()


# In[42]:


# Supprimer les valeurs nulls, nous avons 126 rows non null avec 10 variables.
data.dropna(inplace=True)
data.head()


# # II. PCA

# In[43]:


# selection des colonnes à prendre en compte dans l'ACP
data_pca = data[['Population','PIB par habitant','Kcal/personne/jour',
                 'disponibilité en protéines origine animales g/hab/j','disponibilité totale en protéines g/hab/j',
                 'Disponibilité intérieure','Importations - Quantité','Production',
                 'Prix Production Volaille (Indice)','Political_Stability']]


# In[44]:


# préparation des données pour l'ACP
data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
X = data_pca.values


# In[45]:


print(X)


# In[46]:


names = data["Zone"]
features = data_pca.columns


# In[47]:


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
print(X_scaled)


# In[48]:


# Calcul des composantes principales
pca = decomposition.PCA()
pca.fit(X_scaled)


# In[49]:


# Select the Best Number of Principal Components for the Dataset

import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_, marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")

# D'apres les resultats, avec 3 composants nous arrivons a avoir 71% d'inertie, qui est suffisant.


# In[50]:


#Pourcentage de variance expliquée
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum()) 


# In[51]:


# choix du nombre de composantes
n_comp = 3


# In[52]:


# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(1,2),(0,2)], labels = np.array(features))


# Dimension & Caracteristique & Pays contributeur
# 
# F1: Richesse de pays - Pays économiquement stable - UK & Luxembourg
# 
# F2: taille de pays - plus c'est grand plus il y a la disponibilite interieure - Chine & Inde
# 
# F3: Prix de la Production Volaille - cout de la production - Japon & UK

# In[53]:


# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(1,2), (0, 2)], labels = np.array(names))


# In[54]:


# Matrice de correlation entre variables
corr = data_pca.corr()

# Set up the matplotlib plot configuration

f, ax = plt.subplots(figsize=(12, 10))

# Configure a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap

sns.heatmap(corr, annot=True, cmap=cmap)
ax.set_title('Matrice de correlation entre variables')


# # III. Classification ascendante hiérarchique 

# In[55]:


from scipy.cluster.hierarchy import dendrogram
from functions import plot_dendrogram

# Clustering hiérarchique methode Ward
Z = linkage(X_scaled, 'ward')

# Affichage du dendrogramme
plot_dendrogram(Z, list(names))

# Nous avons 3 clusters d'apres le dendrogram


# In[56]:


# Clustering hiérarchique methode Single
Z1 = linkage(X_scaled, 'single')

# Affichage du dendrogramme
plot_dendrogram(Z1, list(names))


# In[57]:


# Clustering hiérarchique methode Complete
Z1 = linkage(X_scaled, 'complete')

# Affichage du dendrogramme
plot_dendrogram(Z1, list(names))


# # IV. Projetection des groupes de pays dans le ACP

# In[58]:


# Nous souhaitons avoir 3 clusters
clusters = fcluster(Z, t=3, criterion='maxclust')
clusters


# In[59]:


# Projection des groupe
X_projected = pca.transform(X_scaled)

display_factorial_planes(X_projected, n_comp, pca, [(0,1),(1,2), (0,2)], clusters=clusters,  labels = np.array(names))


# In[60]:


# Projection des groupe
X_projected = pca.transform(X_scaled)

display_factorial_planes(X_projected, n_comp, pca, [(0,1),(1,2), (0,2)], clusters=clusters)


# # V. K-Means & centroïdes

# In[61]:


# Definir le K de K-means avec silouette
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(X_scaled)
  labels = kmeans.labels_
  sil.append(silhouette_score(X_scaled, labels, metric = 'euclidean'))

plt.plot(list(range(2, kmax+1)), sil)
plt.show()


# In[62]:


# On instancie notre Kmeans avec 3 clusters : 
kmeans = KMeans(n_clusters=3)

# On l'entraine : 
kmeans.fit(X_scaled)

# On peut stocker nos clusters dans une variable labels : 
labels_kmeans = kmeans.labels_


# In[63]:


km = KMeans(n_clusters=3).fit(X_scaled)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.Zone
cluster_map['cluster'] = km.labels_


# In[64]:


# Liste des pays dans le cluster 1
cluster_map[cluster_map.cluster == 0]


# In[65]:


# Liste des pays dans le cluster 2
cluster_map[cluster_map.cluster == 1]


# In[66]:


# Liste des pays dans le cluster 3
cluster_map[cluster_map.cluster == 2]


# In[67]:


# On peut stocker nos centroids dans une variable : 
centroids = kmeans.cluster_centers_
centroids


# In[68]:


# On utilise bien le scaler déjà entrainé : 

centroids_proj = pca.transform(centroids)


# In[69]:


# On définit notre figure et son axe : 
fig, ax = plt.subplots(1,1, figsize=(8,7))

# On affiche nos individus, avec une transparence de 50% (alpha=0.5) : 
ax.scatter(X_projected[:, 0], X_projected[:, 1], c=labels_kmeans, cmap="Set1", alpha =0.5)


# On affiche nos centroides, avec une couleur noire (c="black") et une frome de carré (marker="c") : 
ax.scatter(centroids_proj[:, 0], centroids_proj[:, 1],  marker="s", c="black" )

# Nous avons 2 graphiques des clusters avec K-means et avec CSH pour faire la comparaison
ax.set_xlabel("F1")
ax.set_ylabel("F2")
plt.title("K-means")

plt.show()


fig, ax = plt.subplots(1,1, figsize=(8,7))
ax.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters, cmap="Set1", alpha =0.5)
plt.title("Classification ascendante hiérarchique")
ax.set_xlabel("F1")
ax.set_ylabel("F2")
plt.show()


# ### En comparant les groupes generes avec les 2 methodes, les resultats sont tres similaires.

# # VII. Heatmap avec les clusters et les différentes variables

# In[70]:


data_pca['cluster'] = km.labels_


# In[71]:


data_pca.head()


# In[72]:


# create a dataframe with les clusters en moyenne et les différentes variables
d = {}
index = []

for f in features:
    index.append(f)
    
for c in sorted(clusters):
    d[c] = []
    
    for f in features:
        x = data_pca[data_pca.cluster == c-1][f].mean()
        d[c].append(x)


df = pd.DataFrame(data=d, index=index)

df


# In[73]:


# normaliser 
df = df.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

# Set up the matplotlib plot configuration

f, ax = plt.subplots(figsize=(12, 10))

# Configure a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap

sns.heatmap(df, annot=True, cmap=cmap)
ax.set_title('Heatmap cluster/moyenne nromalisée des features (centrées & réduites par ligne)')


# # VIII. Preconisation

# Les caracteristiques des groupes:
# 
# Group 1: Politiquement tres stable, besoin d'importation et une taille de pays relativement grand
# Group 2: Economie faible, instable politiquement
# Group 3: pays avec population tres forte, Importation des volailles tres importante
# 
# 
# D'après les résultats, il faudrait sélection groupe 2 selon ces criteres:
# 
# 1. Politiquement stables 
# 2. Prix de production volaille est haut
# 3. Importation des poulets important
# 4. PIB forte donc pouvoir d'achat important
# 
# 
# Exemple:
# Allemagne 	
# France 	
# Japon 	
# Mexique 	
# Pays-Bas 	
# Royaume-Uni de Grande-Bretagne et d'Irlande du nord

# In[ ]:




