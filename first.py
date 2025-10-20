import streamlit as st
import pandas as pd

st.title("学生 小天- 追剧记录📸")
st.header("🔖基础信息")
st.subheader('学生ID:23031310119')
st.markdown(':blue[精神状态:崩溃]')
st.subheader('🎬观看应用:腾讯视频|安全等级:0🔋')
st.markdown('## 🎥剧名')
st.markdown(':green[吴邪私家笔记,藏海花,终极笔记]')
c1, c2, c3 = st.columns(3)
c1.metric(label="原著进度", value="90%", delta="70%")
c2.metric(label="影视剧进度", value="66%", delta="6%")
c3.metric(label="python进度", value='30%', delta="50", delta_color="off")
st.image('C:/Users/26683/OneDrive/桌面/111.jpg',caption='超级好看的一部剧🤗')
st.markdown('## 🎞追剧日历✔️')
import pandas as pd   
import streamlit as st  
data = {
    '日期':['2025-10-1','2025-10-5','2025-10-25'],
    '任务':['吴邪私家笔记','藏海花','终极笔记'],
    '状态':['看完','看差不多','看了一半'],
    '完成程度':['100','80','50']
}
index = pd.Series(['01', '02', '03'], name='剧情')
df = pd.DataFrame(data, index=index)

st.subheader('任务')
st.dataframe(df)

st.subheader('📖')
st.dataframe(df, width=400, height=150)

st.markdown('## 📙代码')
str='''
for i in range(1,10):
    for j in range(1,i+1):
        print(f'{j}*{i}={i*j},end='\t')
    print()
'''
st.code(str,language='python',line_numbers=True)
st.caption('这是一段python代码')

