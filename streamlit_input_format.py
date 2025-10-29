import streamlit as st

island = st.selectbox('企鹅栖息的岛屿', options=['托尔森岛', '比斯科群岛', '德里姆岛'])
sex = st.selectbox('性别', options=['雄性', '雌性'])

bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)
bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)
flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)
body_mass = st.number_input('身体质量（克）', min_value=0.0)

st.write('用户输入的数据是：')
st.text([island, sex, bill_length, bill_depth, flipper_length, body_mass])

#初始化数据预处理格式中与岛屿相关的变量
island_biscoe, island_dream, island_torgerson = 0, 0, 0
#根据用户输入的岛屿数据更改对应的变量值
if island == '比斯科群岛':
    island_biscoe = 1
elif island == '德里姆岛':
    island_dream = 1
elif island == '托尔森岛':
    island_torgerson = 1

#初始化数据预处理格式中与性别相关的变量
sex_female, sex_male = 0, 0
#根据用户输入的性别数据更改对应的变量值
if sex == '雌性':
    sex_female = 1
elif sex == '雄性':
    sex_male = 1

st.write('转换为数据预处理的格式：')
format_data = [bill_length, bill_depth, flipper_length, body_mass,
               island_dream, island_torgerson, island_biscoe, sex_male,
               sex_female]
st.text(format_data)
