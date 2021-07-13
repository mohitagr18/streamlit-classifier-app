import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utilities import *
from classifiers import *

######################## SETUP ########################################

st.set_page_config(page_title="Explore Classifiers", layout="wide")

st.write("""
# Explore Classifiers
""")
st.text("")
st.markdown("""
This simple app allows users to explore the performance of various machine learning classifiers on different datasets.
- Use the menu at left to select the dataset, the classifier, and set model parameters.
""")

######################## DATASET INFO ########################################

# Create dataset names
st.sidebar.subheader("Select dataset and classifier")
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris','Wine','Breast Cancer'))

# Get datatset and class info
X, Y, class_names, descr = get_dataset(dataset_name)

# Get dataset info
shape, columns, null_vals, head = get_data_info(dataset_name)

# Look at the dataset
class_count = pd.DataFrame(Y).iloc[:,0].value_counts()

st.header(f"Let's take a look at the {dataset_name} dataset")
st.text("")
info_col, info_col2 = st.beta_columns(2)

# Details in info_col
info_col.write("Shape of dataset:")
info_col.write(shape)
info_col.write("Number of missing values:")
info_col.write(null_vals)
info_col.write("Features in the dataset:")
info_col.write(columns)

# Details in info_col2
info_col2.write("No. of unique classes:")
info_col2.write(len(np.unique(Y)))
info_col2.write("Class names:")
info_col2.write(class_names)
info_col2.write("Records in each class:")
info_col2.write(class_count)

st.write("Head of the dataset:")
st.write(head)
st.write("Data description:")
with st.beta_expander("Expand to view the description of dataset"):
    st.write(descr)
st.markdown("---")

######################## PLOT DATASET ########################################

# Create Plot
st.header("Let's visualize the data")

st.subheader("Scatter plot: Features colored by class")
st.write("The plots show the variation in each feature colored by class.")
with st.beta_expander("Expand to view the scatter plot"):
    fig, data = create_plot(dataset_name)
    st.pyplot(fig)

st.subheader("Correlation plot")
st.write("The plot shows correlation among numeric features in dataset.")
with st.beta_expander("Expand to view the correlation plot"):
    corr_fig = corr_plot(data)
    st.pyplot(corr_fig)
st.markdown("---")

######################## SIDEBAR UI ########################################

clf_name = add_clf_data(dataset_name)
params = add_parameters(clf_name)

######################## BUILD AND FIT MODEL ########################################

# Split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=100)

# Standardize data
x_train, x_test = standardize_data(clf_name, x_train, x_test)

# Get classifiers
my_clf, skt_clf, param = get_classifiers(clf_name, params)

## Scikit Model ##
acc_skt,precision_skt,recall_skt,cm_skt_fig = skt_clf_fit(skt_clf,x_train,y_train,x_test,y_test)

## Model coded from scratch ##
acc,precision,recall,cm_fig = scratch_clf_fit(clf_name,my_clf,x_train,y_train,x_test,y_test,param)

###### ADDITIONAL MODEL RESULTS #########

# KNN - Plot Error Rate for different k values
if clf_name == "KNN":
    k_fig = find_best_k(x_train, y_train, x_test, y_test, highest_k=15)
# Tree plot
elif clf_name in ["Decision Tree", "Random Forest"]:
    tree_fig = visualize_tree(clf_name, skt_clf, columns, class_names)

######################## MODEL RESULTS UI ########################################

st.header(f"Let's compare the performance of {clf_name} models")
st.text("")
st.markdown("""
Here, we compare the performance of a naive model (coded from scratch) with the model created using `scikit-learn`.
The details of different classifiers coded from scratch can be found __[here](https://mohitagr18.github.io/posts/machinelearning/)__.
""")
mod_col, mod_col2 = st.beta_columns(2)

# Details in mod_col
mod_col.subheader("Model coded from scratch")
mod_col.write(f"Accuracy: {acc}")
mod_col.write(f"Precision: {precision}")
mod_col.write(f"Recall: {recall}")

# Details in mod_col2
mod_col2.subheader("Scikit-learn model")
mod_col2.write(f"Accuracy: {acc_skt}")
mod_col2.write(f"Precision: {precision_skt}")
mod_col2.write(f"Recall: {recall_skt}")

# Confusion Matrix
cm_col1, _, cm_col2, _ = st.beta_columns(4)
cm_col1.write("Confusion Matrix:")
cm_col1.pyplot(cm_fig)
cm_col2.write("Confusion Matrix:")
cm_col2.pyplot(cm_skt_fig)

# Additional plots
if clf_name in ["Decision Tree", "Random Forest"]:
    st.subheader("Visualize tree")
    if clf_name == "Decision Tree":
        st.write("Let's visualize the decision tree created using `scikit-learn` model.")
    else:
        st.write("Let's visualize the __first__ of _n_ decision trees created using `scikit-learn` model.")
    with st.beta_expander("Expand to view the tree"):
        st.write("`scikit-learn` allows the flexibility to visualize a decision tree. Streamlit's conversion of the figure resulting from `tree.plot_tree()` is not great but here is a try.")
        st.pyplot(tree_fig)

if clf_name == "KNN":
    k_col1, _ = st.beta_columns([3,1])
    k_col1.subheader("Best K value")
    k_col1.write("The plot shows error rate for different values of K, generated using model coded from scratch.")
    k_col1.pyplot(k_fig)


######################## END APP ########################################
st.markdown("""
----
""")
