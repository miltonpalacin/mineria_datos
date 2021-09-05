#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.ndimage import gaussian_filter1d
from scipy import signal

# Configuraciones de estilos de Mapplot
# ==============================================================================
plt.style.use('seaborn-whitegrid')
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['savefig.bbox'] = "tight"

# Omitir los Warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Configuracion de impresión de pandas
# ==============================================================================
pd.set_option("display.precision", 8)
pd.set_option('display.width', 1000)


# # Preparación de la señal (en PC)

# In[102]:


get_ipython().system('rm -rf data_original')
get_ipython().system('rm -rf segmentado')
get_ipython().system('mkdir segmentado')
get_ipython().system('mkdir data_original')
get_ipython().system('cp -r emperatriz*/* data_original/')
get_ipython().system('cp -r milca*/* data_original/')
get_ipython().system('cp -r milton*/* data_original/')
get_ipython().system('cp -r raul*/* data_original/')
print("¡Realizado!")


# In[98]:


len(os.listdir("data_original"))


# In[99]:


# nSamplesInSegment: numero de filas que van en un segmento
# nOverlap: numero de filas que se van a superponer
def sampleSegmentation(data, nSamplesInSegment, nOverlap):
    segments = []
    nSamples = len(data)
    for i in range(0, nSamples, nSamplesInSegment-nOverlap):
        if i + nSamplesInSegment > nSamples:
            break
        segments.append(data[slice(i, i + nSamplesInSegment, 1)])
    return segments


def generateSamples(fileName, dir_segmentado):
    numSamples = 100
    numOverlap = 25
    df = pd.read_csv(fileName)
    segments = sampleSegmentation(df, numSamples, numOverlap)
    for i in range(len(segments)):
        f_path_parts = fileName.split("/")
        f_name, f_ext = f_path_parts[-1].split(".")
        segment_file_name = dir_segmentado + f_name + "_" + str(i) + "." + f_ext
        if os.path.exists(segment_file_name):
            os.remove(segment_file_name)
        segments[i].to_csv(segment_file_name)


# ### Ejemplo

# In[5]:


#generateSamples("data_original/emperatriz_caminando_mujer_adulto_20210306122006.csv", 'segmentado/')


# ## Revisando la señal

# In[65]:


def out_gravity(df, per=.5):
    gravity = [0.0, 0.0, 0.0]
    alpha = 0.8
    for index, row in df.iterrows():
        gravity[0] = alpha * gravity[0] + (1 - alpha) * row["x"]
        gravity[1] = alpha * gravity[1] + (1 - alpha) * row["y"]
        gravity[2] = alpha * gravity[2] + (1 - alpha) * row["z"]

        df.loc[index, "x"] = row["x"] - gravity[0]
        df.loc[index, "y"] = row["y"] - gravity[1]
        df.loc[index, "z"] = row["z"] - gravity[2]
    return df[0:int(per*len(df))]


# In[3]:


df_0 = pd.read_csv('data_original/milton_caminando_varon_adulto_20210306083705.csv')
df_1 = pd.read_csv('data_original/milton_sentado_varon_adulto_20210306174604.csv')
df_2 = pd.read_csv('data_original/milton_saltando_varon_adulto_20210306114600.csv')
df_3 = pd.read_csv('data_original/milton_corriendo_varon_adulto_20210324082429.csv')
df_4 = pd.read_csv('data_original/milton_subiendo_varon_adulto_20210324095953.csv')
df_0.shape, df_1.shape, df_2.shape, df_3.shape, df_4.shape


# In[7]:


dx = np.linspace(0, 2000, 250) 

plt.figure(figsize=(25, 8))
plt.plot(dx, df_0.x, 'k', label='Caminando', linestyle="dashed")
plt.plot(dx, df_1.x, 'r', label='Sentado', linestyle="dotted")
plt.plot(dx, df_2.x, 'c', label='Saltando', linestyle="dashed")
plt.plot(dx, df_3.x, 'b', label='Corriendo', linestyle="dashdot")
plt.plot(dx, df_4.x, 'g', label='Subiendo', linestyle="dashed")
plt.xlabel('tiempo $t(s)$', fontsize=18)
plt.ylabel('acelerómetro $(m/s^2)$', fontsize=18)
plt.title('Señal acelerómetro en el x', fontsize=18)
plt.legend()
plt.grid()
#plt.show()
plt.savefig('foo1.png')

plt.figure(figsize=(25, 8))
plt.plot(dx, df_0.y, 'k', label='Caminando', linestyle="dashed")
plt.plot(dx, df_1.y, 'r', label='Sentado', linestyle="dotted")
plt.plot(dx, df_2.y, 'c', label='Saltando', linestyle="dashed")
plt.plot(dx, df_3.y, 'b', label='Corriendo', linestyle="dashdot")
plt.plot(dx, df_4.y, 'g', label='Subiendo', linestyle="dashed")
plt.xlabel('tiempo $t(s)$', fontsize=18)
plt.ylabel('acelerómetro $(m/s^2)$', fontsize=18)
plt.title('Señal acelerómetro en el eje y', fontsize=18)
plt.legend()
plt.grid()
#plt.show()
plt.savefig('foo2.png')
plt.figure(figsize=(25, 8))
plt.plot(dx, df_0.z, 'k', label='Caminando', linestyle="dashed")
plt.plot(dx, df_1.z, 'r', label='Sentado', linestyle="dotted")
plt.plot(dx, df_2.z, 'c', label='Saltando', linestyle="dashed")
plt.plot(dx, df_3.z, 'b', label='Corriendo', linestyle="dashdot")
plt.plot(dx, df_4.z, 'g', label='Subiendo', linestyle="dashed")
plt.xlabel('tiempo $t(s)$', fontsize=18)
plt.ylabel('acelerómetro $(m/s^2)$', fontsize=18)
plt.title('Señal acelerómetro en el z', fontsize=18)
plt.legend()
plt.grid()
#plt.show()

plt.savefig('foo3.png')
plt.show()
# plt.figure(figsize=(25, 8))
# plt.plot(np.linspace(0, 2000, 250), df_0.z, 'k', label='Data Original')
# plt.plot(np.linspace(0, 2000, 250),ss, 'r', label='Data Filtrada')
# plt.legend()
# plt.show()


# ## Aplicando un filtro de suavisado

# In[104]:


df_0 = pd.read_csv('data_original/emperatriz_caminando_mujer_adulto_20210306122006.csv')

ejex = int(1*250)
plt.figure(figsize=(25, 8))
ss = gaussian_filter1d(df_0.z,1.2)

plt.plot(np.linspace(0, 2000, ejex), df_0.z, 'k', label='Data Original')
plt.legend()
plt.show()


plt.figure(figsize=(25, 8))
plt.plot(np.linspace(0, 2000, ejex),ss, 'r', label='Data Filtrada')
plt.legend()
plt.show()

plt.figure(figsize=(25, 8))
plt.plot(np.linspace(0, 2000, ejex), df_0.z, 'k', label='Data Original')
plt.plot(np.linspace(0, 2000, ejex),ss, 'r', label='Data Filtrada')
plt.legend()
plt.show()


# In[105]:


plt.figure(figsize=(25, 8))
fs = len(ss)
# plt.plot(np.array([i for i in range(0, fs)]), data)
plt.plot(np.linspace(0, 5, fs), ss, color="g")
# Leyenda, etiqueta y título
#plt.legend()
#plt.axhline(y=0, color="r", linestyle=":")
#plt.axvline(x=0, color="r", linestyle=":")


# Mostrando el gráfico
plt.show()

spectrum = fft.fft(ss)
freq = fft.fftfreq(len(spectrum))
plt.plot(freq, abs(spectrum))


# In[106]:


plt.figure(figsize=(25, 8))
sos = signal.cheby1(1, 2, 0.3, 'lowpass', output='sos')
lb_x = signal.sosfilt(sos,ss)
lb_x1 = signal.sosfilt(sos,df_0.z)
fs = len(ss)

# plt.plot(np.array([i for i in range(0, fs)]), data)
plt.plot(np.linspace(0, 5, fs), lb_x, color="g")
plt.plot(np.linspace(0, 5, fs), df_0.z, color="b")
plt.plot(np.linspace(0, 5, fs), ss, color="r")
plt.plot(np.linspace(0, 5, fs), lb_x, color="orange")
plt.plot(np.linspace(0, 5, fs), lb_x1, '--',color="c")
# Leyenda, etiqueta y título
#plt.legend()
#plt.axhline(y=0, color="r", linestyle=":")
#plt.axvline(x=0, color="r", linestyle=":")


# Mostrando el gráfico
plt.show()

spectrum = fft.fft(ss)
freq = fft.fftfreq(len(spectrum))
plt.plot(freq, abs(spectrum))


# ## Creación de dataset
# 

# In[107]:


df_0 = out_gravity(pd.read_csv('data_original/emperatriz_caminando_mujer_adulto_20210306122006.csv'))
df_0.head(5)


# In[108]:


def spectral_centroid(x, spect_x):
    magnitudes = np.abs(spect_x)
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length)[:length//2 + 1])
    magnitudes = magnitudes[:length//2 + 1]
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)


# In[109]:


def magnitude(x, y, z):
    ln = len(x)
    sum = 0
    for i in range(ln):
        sum = sum + math.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
    return sum/ln


# In[110]:


dataset_array = []

dir_original_data = 'data_original/'  # carpeta con todos los csv
dir_segmentos = 'segmentado/'  # carpeta donde se almacenaran los nuevos csv creados


# Generando toda las muestras (3 por cada una)
data_original = os.listdir(dir_original_data) 
for mfile in data_original:
    complete_name = dir_original_data + mfile

    if not os.path.isfile(complete_name):
        continue

    generateSamples(complete_name, dir_segmentos)


data_contents = os.listdir(dir_segmentos)
row_id = 0
for mfile in data_contents:
    complete_name = dir_segmentos + mfile

    if not os.path.isfile(complete_name):
        continue
    df_content = pd.read_csv(complete_name)

    lb_x = gaussian_filter1d(df_content.x, 1)
    lb_y = gaussian_filter1d(df_content.y, 1)
    lb_z = gaussian_filter1d(df_content.z, 1)
    #lb_x = df_content.x
    #lb_y = df_content.y
    #lb_z = df_content.z

    # Filtro Digital Pasabajas Chebyshev
    #sos = signal.cheby1(1, 2, 0.3, 'low', output='sos')
    #lb_x = signal.sosfilt(sos, lb_x)
    #lb_y = signal.sosfilt(sos, lb_y)
    #lb_z = signal.sosfilt(sos, lb_z)

    spect_x = abs(fft.fft(lb_x))
    spect_y = abs(fft.fft(lb_y))
    spect_z = abs(fft.fft(lb_z))
    data_class = mfile.split("_")
    class_label = data_class[1]

    spectral_centroid_x = spectral_centroid(lb_x, spect_x)
    spectral_centroid_y = spectral_centroid(lb_y, spect_y)
    spectral_centroid_z = spectral_centroid(lb_z, spect_z)

    magnit = magnitude(lb_x, lb_y, lb_z)
    adm_x = np.average(abs(lb_x - magnit))
    adm_y = np.average(abs(lb_y - magnit))
    adm_z = np.average(abs(lb_z - magnit))

    magnitSpect = magnitude(spect_x, spect_y, spect_z)
    admSpect_x = np.average(abs(spect_x - magnitSpect))
    admSpect_y = np.average(abs(spect_y - magnitSpect))
    admSpect_z = np.average(abs(spect_z - magnitSpect))

    row = []
    row.append(row_id) # id
    row.append(np.mean(lb_x)) # media
    row.append(np.mean(lb_y)) # media
    row.append(np.mean(lb_z)) # media
    row.append(np.mean(spect_x))  # media
    row.append(np.mean(spect_y))  # media
    row.append(np.mean(spect_z))  # media
    row.append(np.median(lb_x)) # mediana
    row.append(np.median(lb_y)) # mediana
    row.append(np.median(lb_z)) # mediana
    row.append(np.median(spect_x)) # mediana
    row.append(np.median(spect_y)) # mediana
    row.append(np.median(spect_z)) # mediana
    row.append(spectral_centroid_x) # spectral centroid
    row.append(spectral_centroid_y) # spectral centroid
    row.append(spectral_centroid_z) # spectral centroid
    row.append(magnit) # magnitude
    row.append(adm_x) # Average Difference from Mean
    row.append(adm_y) # Average Difference from Mean
    row.append(adm_z) # Average Difference from Mean
    row.append(magnitSpect)  # magnitude spect
    row.append(admSpect_x)  # Average Difference from Mean Spect
    row.append(admSpect_y)  # Average Difference from Mean Spect
    row.append(admSpect_z)  # Average Difference from Mean Spect
    row.append(class_label)
    dataset_array.append(row)
    row_id = row_id + 1

cols = ["id", "mean_x", "mean_y", "mean_z", "mean_fft_x", "mean_fft_y", "mean_fft_z", 
        "median_x", "median_y", "median_z", "median_fft_x", "median_fft_y", "median_fft_z", 
        "spec_cent_x", "spec_cent_y", "spec_cent_z", 'magnitude', 
        "adm_x", "adm_y", "adm_z", 
        'magnitudeSpect', "admSpect_x", "admSpect_y", "admSpect_z",
        "class"]

dataset = pd.DataFrame(data=dataset_array, columns=cols)
dataset.head()


# In[111]:


dataset.describe()


# # Modelado / Entrenamiento

# In[112]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.preprocessing import Normalizer
import joblib
import itertools


# ## Read Dataset

# In[113]:


ds = dataset.drop(['id'], axis=1)
ds.shape


# ## Datset Cleaning 

# ### Verificando Outliers

# In[114]:


ds.skew()


# In[115]:


ds.boxplot(figsize=(25, 8))


# In[116]:


zsc = np.abs(stats.zscore(ds.drop(["class"], axis=1), axis=1))
print("Filas que tiene mínimo un outlier en alguna columna:", zsc[(zsc > 2.2)].shape[0])
print("Filas que tiene mínimo un outlier en todas las columnas:", zsc[(zsc > 2.2).all(axis=1)].shape[0])


# In[117]:


Q1 = ds.iloc[:, :-1].quantile(0.25)
Q3 = ds.iloc[:, :-1].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[118]:


ds.iloc[:, :-1] = np.where(ds.iloc[:, :-1] < (Q1 - 1.5 * IQR), Q1, ds.iloc[:, :-1])


# In[119]:


ds.iloc[:, :-1] = np.where(ds.iloc[:, :-1] > (Q3 + 1.5 * IQR), Q3, ds.iloc[:, :-1])


# In[120]:


ds.boxplot(figsize=(25, 8))


# ### Verificando Balanceo de data

# In[121]:


ds.groupby("class", as_index=False).agg({"class": "count"}).rename(columns = {"class":"total_x_class"})


# ##  Data Preprocessing

# In[122]:


ds["class"].value_counts()


# ### Codificar variables categóricas

# In[123]:


ds["class"] = np.where(ds["class"] == "caminando", 0,
                       np.where(ds["class"] == "sentado", 1,
                                np.where(ds["class"] == "saltando", 2,
                                         np.where(ds["class"] == "corriendo", 3, 4
))))


# In[124]:


ds["class"].value_counts()


# ### Normalizar

# In[125]:


#features = ds.columns
X = ds.iloc[:, : -1]
y = ds["class"]


# In[126]:


data_cols = list(X)
scaler = Normalizer().fit(X)
tmp_scaled = scaler.transform(X)
X = pd.DataFrame(tmp_scaled)
X.columns = data_cols
X.describe()


# ### Generando dato de prueba y entranamiento

# In[127]:


#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.5, random_state=2)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 2)


# In[128]:


list(set(y_train))


# ## Modelo, afinamiento

# Entrenamiento con varios parametros

# In[130]:


nodes_per_layer= [30, 40, 50, 60, 70, 80]
learn_rate_init = [10e-6, 10e-5,10e-4, 10e-3]
train_par_res = []
for n_nodes in nodes_per_layer:
    for learn_rate in learn_rate_init:
        clf = MLPClassifier(hidden_layer_sizes=(n_nodes, n_nodes), max_iter=500,
                            learning_rate_init=learn_rate, solver='adam')
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        y_pred = clf.predict(X_test)

        acc_i = accuracy_score(y_test, y_pred)
        vals = []
        vals.append(2) #num hidden layers
        vals.append(n_nodes) #num nodes per layer
        vals.append(clf.n_iter_) #num iterations 
        vals.append(learn_rate)
        vals.append(clf.loss_) #loss
        vals.append(acc_i) #accuracy
        vals.append(train_score) #train_score
        vals.append(test_score) #test_score

        train_par_res.append(vals)

for n_nodes in nodes_per_layer:
    for learn_rate in learn_rate_init:
        clf = MLPClassifier(hidden_layer_sizes=(n_nodes, n_nodes, n_nodes), max_iter=500,
                            learning_rate_init=learn_rate, solver='adam')
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        y_pred = clf.predict(X_test)
        acc_i = accuracy_score(y_test, y_pred)
        vals = []
        vals.append(3) #num hidden layers
        vals.append(n_nodes) #num nodes per layer
        vals.append(clf.n_iter_) #num iterations 
        vals.append(learn_rate)
        vals.append(clf.loss_) #loss
        vals.append(acc_i) #accuracy
        vals.append(train_score) #train_score
        vals.append(test_score) #test_score

        train_par_res.append(vals)

cols = ["Hidden Layers", "Nodes per layer", "# iter", "learning rate", "loss", "accuracy", "train_score", "test_score"]
trains_params_dt = pd.DataFrame(data=train_par_res, columns=cols)
trains_params_dt.head()


# In[131]:


trains_params_dt.sort_values(by=['accuracy'], ascending=False).head(10)


# In[132]:


get_ipython().run_cell_magic('time', '', '# clf.best_loss_, clf.loss_, clf.n_iter_\n#clf = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=300)\nclf = MLPClassifier(hidden_layer_sizes=(70, 70), max_iter=160, learning_rate_init=0.01, solver=\'adam\')\n#clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=200, learning_rate_init=0.001)\nclf.fit(X_train, y_train)\nlr = clf.loss_curve_\nplt.plot(range(len(lr)), lr, color="g")\nclf.score(X_train, y_train)')


# In[133]:


clf.best_loss_, clf.loss_, clf.score(X_train, y_train), clf.score(X_test, y_test)


# In[134]:


filename = 'modelo_deteccion_v4.uni'
joblib.dump(clf, filename)


# In[135]:


y_pred = clf.predict(X_test)


# In[136]:


temp = pd.DataFrame(clf.predict_proba(X_test), columns=["a", "b", "c", "d", "e"])
temp.columns
rg1 = 0.15
rg2 = 0.35

temp[((temp.a < rg2) & (temp.a > rg1)) & ((temp.b < rg2) & (temp.b > rg1))]
# Parece que nunca va encontrar ninguno


# In[137]:


accuracy_score(y_test, y_pred)


# In[138]:


cm = confusion_matrix(y_test.values, y_pred)
cm


# In[139]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '0.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('Etiquetas verdaderas', )
    plt.xlabel('Etiquetas predecidas')


# In[140]:


labels = ['CAMINANDO', 'SENTADO', 'SALTANDO', 'CORRIENDO', 'SUBIENDO']
plot_confusion_matrix(cm, classes=labels, normalize=True, title='Matriz de confusión normalizada', cmap = plt.cm.Greens)


# In[141]:


#= np.where(ds["class"] == "caminando", 0, np.where(ds["class"] == "sentado", 1, 2))
var_tmp = pd.DataFrame(y_test)
var_tmp["pred"] = y_pred
var_tmp["caminando"] = np.where((var_tmp["class"] == var_tmp["pred"]) & (var_tmp["class"] == 0), 1, 0)
var_tmp["sentado"] = np.where((var_tmp["class"] == var_tmp["pred"]) & (var_tmp["class"] == 1), 1, 0)
var_tmp["saltando"] = np.where((var_tmp["class"] == var_tmp["pred"]) & (var_tmp["class"] == 2), 1, 0)
var_tmp["corriendo"] = np.where((var_tmp["class"] == var_tmp["pred"]) & (var_tmp["class"] == 3), 1, 0)
var_tmp["subiendo"] = np.where((var_tmp["class"] == var_tmp["pred"]) & (var_tmp["class"] == 4), 1, 0)
var_tmp


# In[142]:


var_tmp.groupby(["caminando", "sentado", "saltando", "corriendo", "subiendo"], as_index=False).agg({"caminando": "sum", "sentado": "sum", "saltando": "sum", "corriendo": "sum", "subiendo": "sum"})


# In[143]:


cm


# In[144]:


model = joblib.load("modelo_deteccion_v4.uni")


# In[145]:


X_test.iloc[1:2,:]


# In[146]:


y_pred = model.predict(X_test)


# In[147]:


cm = confusion_matrix(y_test.values, y_pred)


# In[148]:


cm


# In[149]:


labels = ['CAMINANDO', 'SENTADO', 'SALTANDO', 'CORRIENDO', 'SUBIENDO']
plot_confusion_matrix(cm, classes=labels, normalize=False, title='Matriz de confusión normalizada', cmap = plt.cm.Greens)


# In[150]:


model.score(X_test, y_test)


# In[ ]:




