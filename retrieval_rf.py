import streamlit as st
import pyterrier as pt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils_rf import get_field_counts_module
from utils_rf import set_result_dic
from utils_rf import get_current_count_table_for_doc
from utils_rf import train_ltr_pipeline
from utils_rf import calc_ltr_preds
from utils_rf import change_r_system

@st.cache_resource
def init(index_path = "/workspace/index/data.properties"):
    if not pt.started():
        pt.init()
    st.session_state['index'] = pt.IndexFactory.of(index_path)
        
@st.cache_resource
def build_database_session(db_conn = "postgresql://user@postgres:5432/", table="webtables"):
    conn_string = db_conn + table
    engine = create_engine(conn_string)
    
    Session = sessionmaker(bind=engine)
    st.session_state['db_session'] = Session()

def set_avail_systems():
    st.session_state['avail_systems'] = ('bm25', 'bm25_BA', 'bm25_Bo1', 'bm25_Bo2', 'bm25_KL', 'bm25_KLComplete', 'bm25_KLCorrect')

def build_resultpage():
    for docno in st.session_state['docnos']:
        current_doc = st.session_state['result_dic'][docno]
        current_table = current_doc['table']
        current_table_title = current_doc['table_title']
        current_page_title = current_doc['page_title']
        current_text_before = current_doc['text_before']
        current_text_after = current_doc['text_after']
        current_score = current_doc['score']
        current_url = current_doc['url']

        tab_table, tab_context, tab_feedback = st.tabs(["Table", "Context", "Feedback"])
        with tab_table:
            if current_page_title:
                st.write(f'Page Title: {current_page_title}')
                
            if current_table_title:
                st.write(f'Table Title: {current_table_title}')

            st.write(current_table)

        with tab_context:
            if current_url:
                st.write(current_url)

            if current_text_before:
                st.write(current_text_before)

            if current_text_after:
                st.write(current_text_after)
        
        with tab_feedback:
        
            col_left, col_right = st.columns(2)
            count_table = get_current_count_table_for_doc(docno)
            
            st.write(count_table)

            with col_left:
                st.write(f'Relevance Score: {round(current_score, 3)}')
                st.button('Relevent', key=f'button_r{docno}', on_click=feedback_handler, args=(docno, True))

            with col_right:
                ltr_score = None
                if docno in st.session_state['ltr_scores']:
                    ltr_score = st.session_state['ltr_scores'][docno]
                
                st.write(f"Feedback Score: {ltr_score}")
                st.button('Not Relevent', key=f'button_nr{docno}', on_click=feedback_handler, args=(docno, False))
                    
                    
            if docno in st.session_state['feedback']:
                st.write(f"Table is relevant: {st.session_state['feedback'][docno]}")

def apply_query(delete_feedback = True, use_ltr_head=False):
    if 'query_str' in st.session_state:
        st.session_state['query'] = st.session_state['query_str']

    if delete_feedback:
        st.session_state['feedback'] = {}

    current_system = st.session_state['r_system']

    if use_ltr_head:
        current_system = current_system >> st.session_state['r_system_head']


    res = current_system.search(st.session_state['query']) 

    st.session_state['scores'] = list(res['score'])[:st.session_state['num_results']]  
    st.session_state['docnos'] = list(res['docno'])[:st.session_state['num_results']]
    st.session_state['count_tables'] = dict(zip(st.session_state['docnos'], list(res['features'])[:st.session_state['num_results']]))

    set_result_dic()

def change_num():
    change_r_system()
    apply_query(delete_feedback=False)

def feedback_handler(docno, relevance):
    st.session_state['feedback'][docno] = relevance

def apply_feedback_handler():
    if st.session_state['feedback']:
        train_ltr_pipeline()
        calc_ltr_preds()

def ltr_rerank_handler():
    apply_query(delete_feedback=False, use_ltr_head=True)

def main():

    # init phase
    if 'index' not in st.session_state:
        print("initialization")
        init()
        st.session_state['feedback'] = {}
        st.session_state['ltr_scores'] = {}
        st.session_state['current_system'] = 'bm25'

    if 'db_session' not in st.session_state:
        build_database_session()
    if 'avail_systems' not in st.session_state:
        set_avail_systems()

    # user inputs
    if not 'query' in st.session_state:
        st.sidebar.text_input('Query', key='query_str', on_change=apply_query)
        st.session_state['query'] = st.session_state['query_str']
    else:
        st.sidebar.text_input('Query', key='query_str', on_change=apply_query, value=st.session_state['query'])

    if not 'num_results' in st.session_state:
        st.sidebar.number_input('Number of results', min_value=1, max_value=50, value=5, key="num_results", on_change=change_num)
    else:
        st.sidebar.number_input('Number of results', min_value=1, max_value=50, value=st.session_state['num_results'], key="num_results", on_change=change_num)


    st.sidebar.selectbox('Retrieval system', st.session_state['avail_systems'], key='current_system_str')
    st.sidebar.button('Apply feedback', on_click=apply_feedback_handler)
    st.sidebar.button('Rerank by feedback', on_click=ltr_rerank_handler)

    if 'current_system_str' in st.session_state:
        st.session_state['current_system'] = st.session_state['current_system_str']

    if 'r_system' not in st.session_state:
        change_r_system()

    if 'query' in st.session_state and 'docnos' in st.session_state:
        print("new result")
        build_resultpage()

if __name__ == "__main__":
    main()