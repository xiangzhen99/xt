import streamlit as st
import pickle
#使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象
with open('rfc_model.pkl', 'rb') as f:
    rfc_model = pickle.load(f)

#使用pickle的load方法从磁盘文件反序列化加载一个之前保存的映射对象
with open('output_uniques.pkl', 'rb') as f:
    output_uniques_map = pickle.load(f)

st.subheader('随机森林模型')
st.write(rfc_model)

st.subheader('映射关系示例')
#“1”应该对应“巴布亚企鹅”
st.write(output_uniques_map[1])
