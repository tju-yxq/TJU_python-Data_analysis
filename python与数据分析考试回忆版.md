# python与数据分析考试回忆版

### （一）选择题（30分,15道）

1. 大数据摩尔定律：人类社会数据**每2年**就会翻一倍

2. python中的三目运算符：min=x if x<y else y

3. 聚类算法：K-Means算法不能直接对文本数据使用

4. 分类算法：KNN的一个距离，不确定是曼哈顿还是欧式距离

5. 分类问题的评估指标：我选的是精确率和召回率，还涉及了ROC曲线

6. 模块与包：选择错误的，我选的是模块不一定是.py文件，主要是我记得课件里面就是用.py文件创建的模块，以下是课件图

![image-20250531170114747](https://github.com/user-attachments/assets/5e0fadf2-86af-4d50-8e4d-232ce793c16f)


7. 负梯度下降算法：这个我是真没见过，但秉着“一定”就是错的原则，选了“负梯度下降算法会使模型一定更优”

8. 回归：选错的，我选了“按因变量个数多少，将回归问题分为线性回归和非线性回归”

9. 集合：应该是将列表导入set()，求结果，我选的是“{1,2,3}”

10. 列表的切片赋值：大概就是下面这个，给一个列表，对后面的赋值

![image-20250531171720205](https://github.com/user-attachments/assets/95f1c917-3969-4cc9-961d-0b01c2e7e0d1)

11. Python控制语句块的方式：缩进
12. 关于函数的判断正误：选正确的，我选的是“函数是为了完成特定任务”
13. 函数无return时会返回什么：None

还有两道是真忘了，一道是判断正误，我记得是选一个错的离谱的“无需创建”，还有一道是选一个关于数据的，我记得我选的是“统计数据”。
### （二）简答题（20分，4道）

1. 大数据的技术特点和生产方式
   
   技术特点：5V（快速化、大量化、真实性、价值密度低，多样化）

   生产方式：日常生活生产

3. python中内置序列的特征

   元组、列表、字典、列表……自由发挥

4. 不可见对象和可见对象的概念，其在函数参数传递时的不同
   
   ![image-20250531180100190](https://github.com/user-attachments/assets/5979823a-8e61-490e-a488-6110977fd1e4)


6. 数据分析的基本流程

   ![image-20250531165704909](https://github.com/user-attachments/assets/311bc1ba-6a96-49dd-958c-5e4f76097327)


### （三）程序填空（30分，15空）

1. 反转字符串：leetcode第541题反转字符串为相同的题干，但题目内容不同，要看长的版本
   
   印象中的答案：left+=1，right-=1，reverse(i，j)，i+k-1，2k

3. 房价预测：线性回归课件中的改编，加了一个pandas的独热编码（很美了）

   印象中的答案："建筑年份"，get_dummies()，mean，mean_houses，fit_transfomer，fit，predict，train_test_split，drop，y_test，y_pred，coef_ 

### （四）综合分析题（20分，两道）

1. 一道文科题，讨论大数据技术造福了哪些行业，说说“从IT时代迈向DT时代”这句话的理解
2. 说一说维度灾难的定义、解决方案；说一说模型复杂度、过拟合、欠拟合的联系和三者的概念。
