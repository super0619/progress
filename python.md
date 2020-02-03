```python
cal=A.sum(axis=0)//列相加
cal=A.sum(axis=1)//行相加
percentage=A/(cal.reshape(1,4))//reshape确保cal是一个列向量	
```

矩阵的加减乘除会转换为对应元素之间的运算