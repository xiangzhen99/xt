import streamlit as st
import pickle
import pandas as pd

# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="企鹅分类器",  
    page_icon=":penguin:",  
    layout='wide',
)

# 使用侧边栏实现多页面显示效果
with st.sidebar:
    st.image('images/rigth_logo.png', width=100)
    st.title('请选择页面')
    page = st.selectbox("请选择页面", ["简介页面", "预测分类页面"], label_visibility='collapsed')

if page == "简介页面":
    st.title("企鹅分类器:penguin:")
    st.header('数据集介绍')
    st.markdown("""帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，也可以作为机器学习入门练习。
该数据集是由Gorman等收集，并发布在一个名为palmerpenguins的R语言包，以对南极企鹅种类进行分类和研究。
该数据集记录了344行观测数据，包含3个不同物种的企鹅：阿德利企鹅、巴布亚企鹅和帽带企鹅的各种信息。""")
    st.header('三种企鹅的卡通图像')
    st.image('images/penguins.png')

elif page == "预测分类页面":
    st.header("预测企鹅分类")
    st.markdown("这个Web应用是基于帕尔默群岛企鹅数据集构建的模型。请输入以下信息开始预测！")
    # 该页面是3:1:2的列布局
    col_form, col, col_logo = st.columns([3, 1, 2])
    with col_form:
        # 运用表单和表单提交按钮
        with st.form('user_inputs'):
            island = st.selectbox('企鹅栖息的岛屿', options=['托尔斯森岛', '比斯科群岛', '德里姆岛'])
            sex = st.selectbox('性别', options=['雄性', '雌性'])
            
            bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)
            bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)
            flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)
            body_mass = st.number_input('身体质量（克）', min_value=0.0)
            
            submitted = st.form_submit_button('预测分类')

    # 初始化数据预处理格式中与岛屿相关的变量
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    # 根据用户输入的岛屿数据更改对应的值
    if island == '比斯科群岛':
        island_biscoe = 1
    elif island == '德里姆岛':
        island_dream = 1
    elif island == '托尔斯森岛':
        island_torgerson = 1

    # 初始化数据预处理格式中与性别相关的变量
    sex_female, sex_male = 0, 0
    # 根据用户输入的性别数据更改对应的值
    if sex == '雌性':
        sex_female = 1
    elif sex == '雄性':
        sex_male = 1

    # 核心修复：严格保留模型需要的9个特征（与模型训练时的特征顺序完全一致）
    format_data = [
        bill_length,         # 喙的长度
        bill_depth,          # 喙的深度
        flipper_length,      # 翅膀的长度
        body_mass,           # 身体质量
        island_dream,        # 岛屿_德里姆岛
        island_torgerson,    # 岛屿_托尔斯森岛
        island_biscoe,       # 岛屿_比斯科群岛
        sex_male,            # 性别_雄性
        sex_female           # 性别_雌性
    ]

    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象
    with open('rfc_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)

    # 加载物种映射对象（用于将模型输出的编码转为物种名称）
    with open('output_uniques.pkl', 'rb') as f:
        output_uniques_map = pickle.load(f)

    # 当用户点击“预测分类”按钮后执行预测逻辑
    if submitted:
        # 将用户输入的格式数据转为DataFrame，列名匹配模型的特征名
        format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)
        # 使用模型预测，得到物种的类别代码（提取标量值，解决ndarray不可哈希问题）
        predict_result_code = rfc_model.predict(format_data_df)[0]
        # 将类别代码映射为具体的物种名称
        predict_result_species = output_uniques_map[predict_result_code]
        
        # 在页面显示预测结果
        st.write(f'根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**')

    # 右侧Logo区域的显示逻辑
    with col_logo:
        # 未提交预测时，显示默认Logo
        if not submitted:
            st.image('images/rigth_logo.png', width=300)
        # 提交预测后，显示对应物种的图片
        elif submitted:
            st.image(f'images/{predict_result_species}.png', width=300)
