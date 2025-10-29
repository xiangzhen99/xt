import pandas as pd

pd.set_option('display.unicode.east_asian_width', True)
#读取数据集，并将字符编码指定为gbk，防止中文报错
penguin_df = pd.read_csv('penguins-chinese.csv', encoding='gbk')
#删除缺失值所在的行
penguin_df.dropna(inplace=True)
#将企鹅的种类定义为目标输出变量
output = penguin_df['企鹅的种类']
#将去除年份列不作为特征列
#使用企鹅栖息的岛屿、喙的长度、翅膀的长度、身体质量、性别作为特征列
features = penguin_df[['企鹅栖息的岛屿', '喙的长度', '喙的深度', '翅膀的长度', '身体质量', '性别']]
#对特征列进行独热编码
features = pd.get_dummies(features)
#将目标输出变量转换为离散数值
output_codes, output_uniques = pd.factorize(output)

print('下面是去重后，目标输出变量的数据：')
print(output_uniques)
print('下面是独热编码后，特征列的数据：')
print(features.head())
