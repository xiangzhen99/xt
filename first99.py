import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# ----------------------1. 全局配置与初始化----------------------
# 页面基础设置
st.set_page_config(
    page_title="企鹅分类器",
    page_icon=":penguin:",
    layout="wide"
)

# 定义文件路径（确保数据集与代码同目录）
DATA_PATH = "penguins-chinese.csv"
MODEL_PATH = "rfc_model.pkl"
MAP_PATH = "output_uniques.pkl"
FEATURE_PATH = "feature_names.pkl"

# 图片路径（需在代码同目录创建images文件夹并放入对应图片）
LOGO_PATH = "images/rigth_logo.png"
PENGUINS_SUMMARY_PATH = "images/penguins.png"
SPECIES_IMAGE_PATHS = {
    "阿德利企鹅": "images/阿德利企鹅.png",
    "巴布亚企鹅": "images/巴布亚企鹅.png",
    "帽带企鹅": "images/帽带企鹅.png"
}

# ----------------------2. 数据预处理与模型训练工具函数----------------------
def preprocess_data(df):
    """数据预处理：处理缺失值、编码分类特征"""
    # 1. 修复列名与数据错位（原数据"喙的长度"列混入性别数据，重新定义有效列）
    correct_columns = ["企鹅的种类", "企鹅栖息的岛屿", "性别", "喙的深度", "翅膀的长度", "身体质量"]
    df_clean = df[correct_columns].copy()

    # 2. 处理缺失值
    # 数值特征（喙的深度、翅膀的长度、身体质量）用中位数填充
    num_features = ["喙的深度", "翅膀的长度", "身体质量"]
    num_imputer = SimpleImputer(strategy="median")
    df_clean[num_features] = num_imputer.fit_transform(df_clean[num_features])

    # 分类特征（岛屿、性别）用众数填充
    cat_features = ["企鹅栖息的岛屿", "性别"]
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_clean[cat_features] = cat_imputer.fit_transform(df_clean[cat_features])

    # 3. 编码分类特征
    # 岛屿：One-Hot编码（drop='first'避免多重共线性）
    encoder_island = OneHotEncoder(sparse_output=False, drop="first")
    island_encoded = encoder_island.fit_transform(df_clean[["企鹅栖息的岛屿"]])
    island_df = pd.DataFrame(
        island_encoded,
        columns=[f"岛屿_{cat}" for cat in encoder_island.categories_[0][1:]]  # 列名：岛屿_比斯科群岛、岛屿_德里姆岛
    )

    # 性别：Label编码（雌性=0，雄性=1）
    encoder_sex = LabelEncoder()
    df_clean["性别_编码"] = encoder_sex.fit_transform(df_clean["性别"])

    # 4. 编码目标变量（企鹅种类）
    label_encoder_species = LabelEncoder()
    df_clean["物种_编码"] = label_encoder_species.fit_transform(df_clean["企鹅的种类"])

    # 5. 构建特征矩阵X和目标变量y
    X = pd.concat([
        df_clean[num_features],
        island_df,
        df_clean[["性别_编码"]]
    ], axis=1)
    y = df_clean["物种_编码"]

    # 定义特征名（用于后续输入匹配）
    feature_names = num_features + list(island_df.columns) + ["性别_编码"]
    X.columns = feature_names

    # 返回预处理结果与编码器
    return X, y, feature_names, label_encoder_species, encoder_island, encoder_sex

def train_and_save_model():
    """训练随机森林模型并保存（首次运行或模型文件缺失时调用）"""
    # 1. 加载数据集（关键改动：指定编码为'gbk'，解决UnicodeDecodeError）
    if not os.path.exists(DATA_PATH):
        st.error(f"数据集文件未找到！请确保{DATA_PATH}与代码在同一目录")
        st.stop()
    df = pd.read_csv(DATA_PATH, encoding='gbk')

    # 2. 数据预处理
    X, y, feature_names, label_encoder_species, _, _ = preprocess_data(df)

    # 3. 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 训练随机森林模型
    rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc_model.fit(X_train, y_train)

    # 5. 验证模型（显示准确率）
    train_acc = rfc_model.score(X_train, y_train)
    test_acc = rfc_model.score(X_test, y_test)
    st.success(f"模型训练完成！训练准确率：{train_acc:.2f}，测试准确率：{test_acc:.2f}")

    # 6. 保存模型、特征名与物种映射
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rfc_model, f)
    
    with open(FEATURE_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    
    # 物种编码映射（数字→中文名称）
    species_map = dict(zip(
        label_encoder_species.transform(label_encoder_species.classes_),
        label_encoder_species.classes_
    ))
    # 适配原代码中的映射格式（{0: [物种名], 1: [物种名], ...}）
    output_uniques_map = {k: [v] for k, v in species_map.items()}
    with open(MAP_PATH, "wb") as f:
        pickle.dump(output_uniques_map, f)

    return rfc_model, output_uniques_map, feature_names

def load_model_or_train():
    """加载已保存的模型，若模型不存在则自动训练"""
    if os.path.exists(MODEL_PATH) and os.path.exists(MAP_PATH) and os.path.exists(FEATURE_PATH):
        # 加载已保存的模型与映射
        with open(MODEL_PATH, "rb") as f:
            rfc_model = pickle.load(f)
        with open(MAP_PATH, "rb") as f:
            output_uniques_map = pickle.load(f)
        with open(FEATURE_PATH, "rb") as f:
            feature_names = pickle.load(f)
        return rfc_model, output_uniques_map, feature_names
    else:
        # 模型不存在，自动训练
        st.info("未检测到预训练模型，正在自动训练...")
        return train_and_save_model()

# ----------------------3. 用户输入处理函数（核心修复：严格匹配特征名称与编码）----------------------
def process_user_input(island, sex, bill_depth, flipper_length, body_mass, feature_names):
    """将用户输入转换为模型可接受的特征格式（确保特征名称与模型训练时完全一致）"""
    # 1. 编码岛屿（与训练时的One-Hot编码完全对齐）
    island_biscoe = 1 if island == "比斯科群岛" else 0
    island_dream = 1 if island == "德里姆岛" else 0
    # 注意：托尔森岛是One-Hot编码的基准类别（drop='first'时被排除），无需显式编码

    # 2. 编码性别（雌性=0，雄性=1）
    sex_encoded = 1 if sex == "雄性" else 0

    # 3. 构建特征列表（严格匹配模型训练时的特征顺序和名称）
    input_data = [
        bill_depth,          # 喙的深度
        flipper_length,      # 翅膀的长度
        body_mass,           # 身体质量
        island_biscoe,       # 岛屿_比斯科群岛
        island_dream,        # 岛屿_德里姆岛
        sex_encoded          # 性别_编码
    ]

    # 4. 转换为DataFrame（确保特征名与模型训练时完全匹配）
    input_df = pd.DataFrame([input_data], columns=feature_names)
    return input_df

# ----------------------4. 页面内容与交互逻辑----------------------
def main():
    # 加载模型（首次运行自动训练）
    rfc_model, output_uniques_map, feature_names = load_model_or_train()

    # 侧边栏页面选择
    with st.sidebar:
        # 显示侧边栏Logo（若文件不存在显示文本提示）
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=100)
        else:
            st.write("🔍 请在images文件夹放入rigth_logo.png")
        
        st.title("请选择页面")
        page = st.selectbox(
            label="页面选择",
            options=["简介页面", "预测分类页面"],
            label_visibility="collapsed"
        )

    # ----------------------5. 简介页面----------------------
    if page == "简介页面":
        st.title("企鹅分类器 :penguin:")

        # 数据集介绍
        st.header("一、数据集介绍")
        st.markdown("""
        帕尔默群岛企鹅数据集是数据科学入门的经典数据集，由Gorman等人收集并发布，包含**344条观测记录**，涵盖南极3种企鹅的核心生物学特征：
        - **阿德利企鹅**：体型较小，广泛分布于托尔森岛、比斯科群岛、德里姆岛
        - **巴布亚企鹅**：体型较大，主要栖息于比斯科群岛
        - **帽带企鹅**：颈部有黑色"帽带"纹路，仅栖息于德里姆岛
        
        数据集核心特征包括：企鹅栖息的岛屿、性别、喙的深度（毫米）、翅膀的长度（毫米）、身体质量（克），可用于物种分类、特征相关性分析等任务。
        """)

        # 三种企鹅图片展示
        st.header("二、三种企鹅对比图")
        if os.path.exists(PENGUINS_SUMMARY_PATH):
            st.image(PENGUINS_SUMMARY_PATH, caption="帕尔默群岛三种企鹅卡通图")
        else:
            st.warning("⚠️ 未找到三种企鹅汇总图（images/penguins.png），请补充图片文件")

    # ----------------------6. 预测分类页面----------------------
    elif page == "预测分类页面":
        st.header("预测企鹅分类")
        st.markdown("""
        基于随机森林模型的企鹅物种预测工具！  
        请输入以下特征（数值参考实际范围：喙深度13-22mm，翅膀长度170-230mm，身体质量2700-6300g），点击"预测分类"获取结果。
        """)

        # 页面布局：3列（表单列:间隔列:结果展示列）
        col_form, col_space, col_result = st.columns([3, 1, 2])

        # 左侧：用户输入表单
        with col_form:
            with st.form("user_input_form"):
                # 1. 选择岛屿（与模型训练时的类别完全一致）
                island = st.selectbox(
                    label="1. 企鹅栖息的岛屿",
                    options=["托尔森岛", "比斯科群岛", "德里姆岛"]
                )

                # 2. 选择性别（与模型训练时的类别完全一致）
                sex = st.selectbox(
                    label="2. 性别",
                    options=["雌性", "雄性"]
                )

                # 3. 喙的深度（毫米）
                bill_depth = st.number_input(
                    label="3. 喙的深度（毫米）",
                    min_value=13.0,
                    max_value=22.0,
                    step=0.1,
                    value=18.0,
                    help="合理范围：13.0-22.0毫米"
                )

                # 4. 翅膀的长度（毫米）
                flipper_length = st.number_input(
                    label="4. 翅膀的长度（毫米）",
                    min_value=170.0,
                    max_value=230.0,
                    step=1.0,
                    value=190.0,
                    help="合理范围：170.0-230.0毫米"
                )

                # 5. 身体质量（克）
                body_mass = st.number_input(
                    label="5. 身体质量（克）",
                    min_value=2700.0,
                    max_value=6300.0,
                    step=50.0,
                    value=3500.0,
                    help="合理范围：2700.0-6300.0克"
                )

                # 提交按钮
                submitted = st.form_submit_button("预测分类", type="primary")

        # 右侧：结果展示（未提交时显示Logo，提交后显示预测结果）
        with col_result:
            if not submitted:
                # 未提交时显示默认Logo
                if os.path.exists(LOGO_PATH):
                    st.image(LOGO_PATH, width=300, caption="请输入特征并点击预测")
                else:
                    st.write("🔍 等待输入特征...")
            else:
                # 处理用户输入（确保特征编码与模型训练时完全对齐）
                input_df = process_user_input(island, sex, bill_depth, flipper_length, body_mass, feature_names)
                
                # 模型预测
                predict_code = rfc_model.predict(input_df)[0]  # 获取预测编码
                predict_species = output_uniques_map[predict_code][0]  # 映射为中文物种名

                # 显示预测结果
                st.success("✅ 预测完成！")
                st.write(f"### 预测物种：**{predict_species}**")

                # 显示对应物种图片
                species_img_path = SPECIES_IMAGE_PATHS.get(predict_species)
                if os.path.exists(species_img_path):
                    st.image(species_img_path, width=300, caption=f"{predict_species}特征图")
                else:
                    st.warning(f"⚠️ 未找到{predict_species}图片，请补充{species_img_path}文件")

                # 显示物种小知识
                species_knowledge = {
                    "阿德利企鹅": "📌 小知识：阿德利企鹅是南极最常见的企鹅，擅长在冰面快速行走，以磷虾和鱼类为食。",
                    "巴布亚企鹅": "📌 小知识：巴布亚企鹅又称'绅士企鹅'，游泳速度可达36公里/小时，是企鹅中游泳最快的物种之一。",
                    "帽带企鹅": "📌 小知识：帽带企鹅因颈部的黑色条纹形似帽带得名，性格较凶猛，常与其他企鹅争夺栖息地。"
                }
                st.markdown(species_knowledge.get(predict_species, ""))

# ----------------------7. 启动应用----------------------
if __name__ == "__main__":
    main()
