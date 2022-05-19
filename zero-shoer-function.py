import numpy as np
import streamlit as st
import pickle
candidate_labels = ['quality', 'price', 'quantity', 'delivery', 'service', 'location']
loaded_model=pickle.load(open('D:/model_deployment/trained_model.sav', 'rb'))

#Create a function
def zero_shot_classifier(input_data):
    # Input data

    prediction = loaded_model(input_data, candidate_labels, multi_label=False)
    # printout of the prediction
    print(prediction)

    # Visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    # convert scores to percentages
    for i, x in enumerate(prediction['scores']):
        prediction['scores'][i] = x * 100

    plt.figure(figsize=(12, 4))
    sns.barplot(x=prediction['scores'], y=prediction['labels'], orient='horizontal')
    plt.title('Classification Representation')
    plt.xlabel('Percentage')
    plt.ylabel('Class')
    return plt.show()

#Main function
def main():

    #giving title
    st.title('Question classifier')
    #Getting input data from the user
    question=st.text_input('enter a question')
    #code for prediction
    prediction=''
    #Creating a button
    if st.button('classifier'):
        prediction=zero_shot_classifier(question)
    st.success(prediction)

if __name__=='__main__':
    main()


