
#第8章/streamlit_predict.py
import streamlit as st
import pickle

island = st.selectbox('企鹅栖息的岛屿', options=['托尔森岛', '比斯科群岛', '德里姆岛'])
sex = st.selectbox('性别', options=['雄性', '雌性'])

bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)
bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)




flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)
body_mass = st.number_input('身体质量（克）', min_value=0.0)

#初始化数据预处理格式中与岛屿相关的变量
island_biscoe, island_dream, island_torgerson = 0, 0, 0
#根据用户输入的岛屿数据更改对应的值
if island == '比斯科群岛':
    island_biscoe = 1
elif island == '德里姆岛':
    island_dream = 1
elif island == '托尔森岛':
    island_torgerson = 1

#初始化数据预处理格式中与性别相关的变量
sex_female, sex_male = 0, 0
#根据用户输入的性别数据更改对应的值
if sex == '雌性':
    sex_female = 1
elif sex == '雄性':
    sex_male = 1

format_data = [bill_length, bill_depth, flipper_length, body_mass,
               island_dream, island_torgerson, island_biscoe, sex_male,
               sex_female]

#使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象
with open('rfc_model.pkl', 'rb') as f:
    rfc_model = pickle.load(f)

#使用pickle的load方法从磁盘文件反序列化加载一个之前保存的映射对象
with open('output_uniques.pkl', 'rb') as f:
    output_uniques_map = pickle.load(f)

#使用模型对格式化后的数据format_data进行预测，返回预测的类别代码
predict_result_code = rfc_model.predict([format_data])
#将类别代码映射到具体的类别名称
predict_result_species = output_uniques_map[predict_result_code][0]

st.write('根据您输入的数据，预测该企鹅的物种名称是：', predict_result_species)
