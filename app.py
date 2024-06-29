import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
import joblib

# Load your data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Perform BERTopic modeling
def create_model(texts, num_topics, min_topic_size, nr_topics, n_gram_range=(1,3)):
    model = BERTopic(n_gram_range=n_gram_range, calculate_probabilities=True, nr_topics=num_topics, min_topic_size=min_topic_size)
    topics, probs = model.fit_transform(texts)

    topic_info = model.get_topic_info()
    
    st.write(topic_info.head(nr_topics))  # Display top topics based on nr_topics

    # Ensure topic embeddings exist before visualization
    if hasattr(model, 'topic_embeddings_') and model.topic_embeddings_ is not None:
        try:
            fig = model.visualize_topics()
            # Display the figure in Streamlit
            st.plotly_chart(fig)
        except IndexError as e:
            st.error(f"IndexError: {e}")
            st.error("An error occurred during visualization. Please check the topic embeddings and indices.")
    else:
        st.error("Topic embeddings not found, skipping visualization.")

    joblib.dump(model, 'bertopic_model.pkl')
    return topics, topic_info

# Generate topics for the data
def generate_topics(texts, num_topics, min_topic_size, nr_topics):
    n_gram_range = (1, 3)  # Adjust the n-gram range if necessary
    topics, topic_info = create_model(texts, num_topics, min_topic_size, nr_topics, n_gram_range)
    return topics, topic_info

# Function to assign categories based on the cause of anger
def assign_category(description):
    description = description.lower()
    if 'mother' in description or 'father' in description or 'sister' in description or 'brother' in description:
        return 'Family'
    elif 'boyfriend' in description or 'girlfriend' in description or 'couple' in description:
        return 'Relationship'
    elif 'angry' in description or 'argument' in description or 'angry' in description:
        return 'Friendship (Conflict)'
    elif 'fight' in description or 'punch' in description or 'friend' in description or 'hurt' in description:
        return 'Friendship (Fight)'
    elif 'betrayed' in description or 'anger' in description:
        return 'Friendship (Betrayal)'
    elif 'exam' in description or 'work' in description:
        return 'Work'
    else:
        return 'Other'

# Streamlit UI
st.title('BERTOPIC TOPIC MODELING')

# Input parameter to setup BERTopic
st.write("Please provide BERTopic parameters as follows:")

st.write('Note:')
st.info("USER INPUT : You can adjust the parameters such as the number of topics, minimum topic size, and number of top words to fine-tune the BERTopic model according to your specific data and analysis needs. Note that the default parameters are 9, 5 and 12.", icon="ℹ️")

# Number of Topics
num_topics = st.number_input("Number of Topics", min_value=2, max_value=50, value=9)
# Min Topic Size
min_topic_size = st.number_input("Minimum Topic Size", min_value=2, max_value=50, value=5)
# Nr of Top Words
nr_topics = st.number_input("Number of Top Words", min_value=2, max_value=30, value=12)

st.write('Note:')
st.info("USER INPUT : Please copy all the text in the txt file that we have provided, and paste it in textbox below. The text should be more than 100 text otherwise it cannot proceed to process the topics.", icon="ℹ️")

# Text input
st.write("Please input your text (one per line):")
user_input = st.text_area("Input text", height=300)
texts = user_input.split("\n")

if user_input:
    data = pd.DataFrame({"text": texts})
    st.write("Sample data:")
    st.write(data.head())

    # Assign categories based on the cause of anger
    data['category'] = data['text'].apply(assign_category)

    topics, topic_info = generate_topics(texts, num_topics, min_topic_size, nr_topics)
    data['topic'] = topics

    # Calculate predominant category for each topic
    predominant_categories = data.groupby('topic')['category'].agg(lambda x: x.value_counts().idxmax())
    topic_info['category'] = topic_info['Topic'].map(predominant_categories)

    # Merge the topic information with the main dataframe
    topic_info = topic_info.rename(columns={"Name": "topic_name"})
    data_with_info = data.merge(topic_info[['Topic', 'topic_name', 'category']], left_on='topic', right_on='Topic', how='left')

    # Drop the redundant 'Topic' column
    data_with_info = data_with_info.drop(columns=['Topic'])
    
    st.write("Data with topics and categories:")
    st.write(data_with_info)

    st.write("Topic Info with Categories:")
    st.write(topic_info)
