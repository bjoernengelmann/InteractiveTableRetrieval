import pyterrier as pt
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Text, ARRAY

import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter

from sklearn.ensemble import RandomForestRegressor

domain_expression = "^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)"

Base = declarative_base()

class Table(Base):
    __tablename__ = 'wts2'
    docno = Column(String, primary_key=True)
    table_content = Column(Text)
    textBefore = Column(Text)
    textAfter = Column(Text)
    pageTitle = Column(String)
    title = Column(String)
    entities = Column(Text)
    url = Column(String)
    orientation = Column(String)
    header = Column(String)
    key_col = Column(String)
    relation = Column(ARRAY(String, dimensions=2))
    
    def __repr__(self):
        repr_str = f"docno={self.docno}, table_content={self.table_content}, textBefore={self.textBefore}, textAfter={self.textAfter},"\
        f"pageTitle={self.pageTitle}, title={self.title}, entities={self.entities}, url={self.url}, orientation={self.orientation},"\
        f"header={self.header}, key_col={self.key_col}"
        
        return repr_str

def term_in_field(row):
    instance = st.session_state['db_session'].query(Table).filter(Table.docno == row['docno']).first()
    rows = list(map(list, zip(*instance.relation)))
    header = " ".join(rows[0])
    table = " ".join([" ".join(rows[i]) for i in range(1, len(rows))])

    fields = [instance.pageTitle, instance.title , header, table, instance.textBefore, instance.textAfter]
    terms = st.session_state['query'].split(" ")
    result = np.array([[field.lower().count(term.lower()) for field in fields] for term in terms])
    return result

def domain(row):
    instance = st.session_state['db_session'].query(Table).filter(Table.docno == row['docno']).first()
    url = instance.url
    domain = re.findall(domain_expression, url)
    if domain:
        domain = domain[0]
    return domain

def get_field_counts_module():
    field_counts = pt.apply.doc_features(lambda row: term_in_field(row))
    return field_counts

def get_filter_factor(row):
    domain_black_list = []
    filter_fac = 1
    if 'filter_domain_dic' in st.session_state:
        domain_black_list = [key for key, val in st.session_state['filter_domain_dic'].items() if val]

    if row['domain'] in domain_black_list:
        filter_fac = 0

    return filter_fac

def apply_filter_factor(row):
    return row['score'] * row['filter_factor']

def get_domain_module():
    field_counts = pt.apply.domain(lambda row: domain(row))
    filter_factor = pt.apply.filter_factor(lambda row: get_filter_factor(row))
    return field_counts >> filter_factor >> pt.apply.doc_score(lambda row: apply_filter_factor(row))

def set_result_dic():
    docnos = st.session_state['docnos']
    result_dic = {}
    instances = st.session_state['db_session'].query(Table).filter(Table.docno.in_(docnos)).all()
    
    for i in range(len(instances)):
        
        table_array = np.array(instances[i].relation)
        cols, table_array = debuplicate_header([x[0] for x in table_array]), table_array[:,1:]

        table = pd.DataFrame(table_array.T, columns=cols)
        score_index = docnos.index(instances[i].docno)

        entry = {'table' : table, 'page_title' : instances[i].pageTitle,
                 'table_title' : instances[i].title, 'text_before' : instances[i].textBefore,
                 'text_after': instances[i].textAfter, 'score' : st.session_state['scores'][score_index],
                 'url' : instances[i].url}
        
        result_dic[instances[i].docno] = entry

    st.session_state['result_dic'] = result_dic

def debuplicate_header(cols):
    header_list = cols
    counts = {x: header_list.count(x) for x in header_list}
    for i in range(len(header_list)-1, -1, -1):
        if counts[header_list[i]] > 1:
            current_key = header_list[i]
            header_list[i] = header_list[i]+str(counts[header_list[i]])
            counts[current_key] -=1
    
    return header_list

def get_current_count_table_for_doc(docno):
    terms = st.session_state['query'].split(" ")
    field_texts = ["Page title", "Table title" , "Table header", "Table content", "Context before", "Context after"]
    table = st.session_state['count_tables'][docno]
    table = pd.DataFrame(table, columns=field_texts, index=terms)
    return table

def get_current_topics():
    t = {'qid' : 1, 'query' : st.session_state['query']}
    return pd.DataFrame(data=t, index=[0])

def get_current_qrels():
    qrels = []
    for docno, label in st.session_state['feedback'].items():
        qrels.append([1, docno, int(label)])

    qrels = pd.DataFrame(data=np.array(qrels), columns=['qid', 'docno',	'label'])
    return qrels

def train_ltr_pipeline():
    
    pipeline = st.session_state['r_system']
    rf = RandomForestRegressor(n_estimators=400)
    rf_pipe = pipeline >> pt.apply.features(lambda row: row['features'].flatten()) >> pt.ltr.apply_learned_model(rf) 
    train_topics = get_current_topics()
    qrels = get_current_qrels()
    rf_pipe.fit(train_topics, qrels)
    rows_num = len(st.session_state['query'].split(' '))
    
    st.session_state['feature_importance'] = rf.feature_importances_
    r_system_head = pt.apply.features(lambda row: row['features'].flatten()) >> pt.ltr.apply_learned_model(rf) >> pt.apply.features(lambda row: row['features'].reshape((rows_num,6)))
    st.session_state['r_system_head'] = r_system_head >> get_domain_module()
    st.session_state['regresor'] = rf

def calc_ltr_preds():
    regressor = st.session_state['regresor']
    docs = st.session_state['docnos']
    cnt_tables = np.array([st.session_state['count_tables'][docno].flatten() for docno in docs])
    preds = regressor.predict(cnt_tables)
    st.session_state['ltr_scores'] = dict(zip(docs, preds))
    
def get_most_common_domains():
    current_system = st.session_state['r_system']

    if 'r_system_head' in st.session_state:
        current_system = current_system >> st.session_state['r_system_head']
   
    res = current_system.search(st.session_state['query']) 
    docnos = list(res['docno'])
    instances = st.session_state['db_session'].query(Table).filter(Table.docno.in_(docnos)).all()
    urls = [instance.url for instance in instances]
    domains = []
    for url in urls:
        domain = re.findall(domain_expression, url)
        if domain:
            domains.append(domain[0])
        
    return pd.Series(dict(Counter(domains)))

def change_r_system(apply_filter=True):
    print("r_system got changed")
    r_system = None
    if st.session_state['current_system'] == 'bm25':
        r_system = pt.BatchRetrieve(st.session_state['index'], wmodel="BM25")
    
    if apply_filter:
        r_system = r_system >> get_domain_module()

    r_system = r_system >> get_field_counts_module()
    st.session_state['r_system'] = r_system%st.session_state['num_results']
    
def update_filter_list(edited_cells, domains):

    for key, val in edited_cells.items():
        idx = int(key.split(':')[0])
        st.session_state['filter_domain_dic'][domains[idx]] = val

def build_filter_table(total_num_for_stat=1000):
    
    counts = get_most_common_domains().sort_values(inplace=False, ascending=False)
    num_to_display = len(counts.index)

    counts = counts.iloc[:num_to_display]/total_num_for_stat*100
    domain_table = pd.DataFrame()
    domain_table['Percentage'] = counts
    domains = domain_table.index.values.tolist()

    if 'filter_domain_dic' not in st.session_state:
        filter_domain_dic = {}
        for domain in domains:
            filter_domain_dic[domain] = False
        
        st.session_state['filter_domain_dic'] = filter_domain_dic

    domains_to_filter = [key for key, val in st.session_state['filter_domain_dic'].items() if val]
    indices_to_filter = [domain_table.index.get_loc(domains_to_filter[i]) for i in range(len(domains_to_filter))]
    filter_list = [i in indices_to_filter for i in range(num_to_display)]

    domain_table['Filter'] = filter_list
    
    st.experimental_data_editor(domain_table, key='url_filter')
    edited_cells_dic =  st.session_state['url_filter']['edited_cells']
    update_filter_list(edited_cells=edited_cells_dic, domains=domains)
    
