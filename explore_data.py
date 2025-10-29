import pandas as pd

#设置输出右对齐，防止中文不对齐
pd.set_option('display.unicode.east_asian_width', True)
#读取数据集，并将字符编码指定为gbk，防止中文报错
penguin_df = pd.read_csv('penguins-chinese.csv', encoding='gbk')
#输出数据框的前5行
print(penguin_df.head())

