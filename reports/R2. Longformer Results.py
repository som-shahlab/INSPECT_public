#!/usr/bin/env python
# coding: utf-8

# # Analyze Results from LongFormer Model

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# In[ ]:


df = pd.read_csv('./imon_results.csv')

LABELS = ['pe_acute', 'pe_subsegmentalonly', 'pe_positive']
THRESHOLD = 0.5

for l in LABELS: 
    df[f"{l}_pred"] = df[f"{l}_prob"].apply(lambda x: 1 if x >= THRESHOLD else 0)
df


# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(figsize=(7, 7))
lw = 2

COLORS = [
    'darkorange', 'lightgreen', 'navy', 'black', 'purple'
]


for idx, labels in enumerate(LABELS):

    label = df[labels].to_list()
    prob = df[f"{labels}_prob"].to_list()
    fpr_v, tpr_v, _ = metrics.roc_curve(label, prob)
    roc_auc_v = metrics.auc(fpr_v, tpr_v)
    plt.plot(
        fpr_v, tpr_v, color=COLORS[idx], alpha=0.5,
        lw=lw, 
        label=f'{labels} ROC curve (area = %0.4f)' % roc_auc_v
    )
'''
labels = 'positive_pe'
label = df[labels].to_list()
prob = df[f"{labels}_prob"].to_list()
fpr_v, tpr_v, _ = metrics.roc_curve(label, prob)
roc_auc_v = metrics.auc(fpr_v, tpr_v)
plt.plot(
    fpr_v, tpr_v, color=COLORS[idx],
    lw=lw, 
    label=f'{labels} ROC curve (area = %0.4f)' % roc_auc_v
)
'''

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 0.95])
#plt.ylim([0.0, 1.05])
#plt.axes().set_aspect('equal', 'datalim')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC')
plt.legend(loc="lower right")

plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

# style
plt.clf()
plt.style.use('ggplot')
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
f, axes = plt.subplots(1, len(LABELS), sharey=True, figsize=(21,6), dpi=300)
#figure(num=None, figsize=(8, 18), dpi=300, facecolor='w', edgecolor='k')

for l, ax in zip(LABELS, axes):
    # Vision
    label = df[f"{l}"].to_numpy()
    prob = df[f"{l}_prob"].to_numpy()
    
    negative_probs = prob[label == 0.0]
    positive_probs = prob[label == 1.0]
    bins = np.linspace(0, 1, 30)
    ax.hist([positive_probs, negative_probs], bins, label=['positive','negative'], width=0.01)
    ax.set_title(l, fontsize = 25)



f.tight_layout(pad=0.5)
#plt.hist([positive_probs, negative_probs], bins, label=['positive','negative'], width=0.015)

#plt.hist(positive_probs, bins, label='positive', width=0.01)
plt.legend(loc='upper right')
axes[0].set_xlabel("Predicted Probabilities", fontsize = 25)
axes[0].set_ylabel("Count", fontsize = 25)

plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

plt.style.use('default')

l = LABELS[0]

labels = df[f"{l}"].to_numpy()
pred = df[f"{l}_pred"].to_numpy()

array = confusion_matrix(labels, pred)
total = array.sum(axis=1).reshape(2,-1)
#df_cm = pd.DataFrame(array/total * 100, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
#                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
df_cm = pd.DataFrame(array, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
plt.figure(figsize = (4,4))
plt.title(l)
sn.set(font_scale=1.5)
#annot_kws = {"ha": 'center',"va": 'center'}
ax = sn.heatmap(df_cm, annot=True,linewidths=1, fmt=".1f", cmap=sn.color_palette("Reds"))#, annot_kws=annot_kws)
ax.set_ylim(len(array), 0)

#for t in ax.texts: t.set_text(t.get_text() + " %")


# In[ ]:


from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

plt.style.use('default')

l = LABELS[1]

labels = df[f"{l}"].to_numpy()
pred = df[f"{l}_pred"].to_numpy()

array = confusion_matrix(labels, pred)
total = array.sum(axis=1).reshape(2,-1)
#df_cm = pd.DataFrame(array/total * 100, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
#                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
df_cm = pd.DataFrame(array, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
plt.figure(figsize = (4,4))
plt.title(l)
sn.set(font_scale=1.5)
#annot_kws = {"ha": 'center',"va": 'center'}
ax = sn.heatmap(df_cm, annot=True,linewidths=1, fmt=".1f", cmap=sn.color_palette("Reds"))#, annot_kws=annot_kws)
ax.set_ylim(len(array), 0)

#for t in ax.texts: t.set_text(t.get_text() + " %")


# In[ ]:


LABELS


# In[ ]:


from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

plt.style.use('default')

l = LABELS[2]

labels = df[f"{l}"].to_numpy()
pred = df[f"{l}_pred"].to_numpy()

array = confusion_matrix(labels, pred)
total = array.sum(axis=1).reshape(2,-1)
#df_cm = pd.DataFrame(array/total * 100, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
#                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
df_cm = pd.DataFrame(array, index = [i for i in [" Actual\n Negative", "Actual\n Positive"]],
                  columns = [i for i in ["Predicted \nNegative", "Predicted \nPositive"]])
plt.figure(figsize = (4,4))
plt.title(l)
sn.set(font_scale=1.5)
#annot_kws = {"ha": 'center',"va": 'center'}
ax = sn.heatmap(df_cm, annot=True,linewidths=1, fmt=".1f", cmap=sn.color_palette("Reds"))#, annot_kws=annot_kws)
ax.set_ylim(len(array), 0)

#for t in ax.texts: t.set_text(t.get_text() + " %")

