import streamlit as st
import pandas as pd

from utils_rf import change_r_system
from utils_rf import build_filter_table
from retrieval_rf import apply_query

col_left, col_right = st.columns(2)

with col_left:
    st.header("Explanations for LTR")

    if "feature_importance" in st.session_state:
        terms = st.session_state['query'].split(" ")
        
        field_texts = ["Page title", "Table title" , "Table header", "Table content", "Context before", "Context after"]
        rows_num = len(terms)
        table = pd.DataFrame(st.session_state['feature_importance'].reshape((rows_num,6)), columns=field_texts, index=terms)

        st.write(table)
    else:
        st.write("No feedback has been provided yet")

with col_right:

    st.header("Frequent domains")


    if 'r_system' in st.session_state:
        
        num_results = 1000
        current_r_system = st.session_state['r_system']
        current_num_results = st.session_state['num_results']
        st.session_state['num_results'] = num_results
        change_r_system()
            
        build_filter_table(num_results)
        st.session_state['num_results'] = current_num_results
        apply_query()
    else:
        st.write("No query has been applied yet")


#st.session_state