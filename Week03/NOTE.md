学习笔记
# 一、学习内容
## 一、递归
递归本质是循环，通过函数体来进行的循环  
### 递归模板
```python
def recursion(level, param1, param2, ...):
    # 递归终止条件
    if level > MAX_LEVEL:
        process_result
        return
    # 处理当前层逻辑
    process(level, data ...)
    # 进入下一层
    recursion(level + 1, p1, ...)
    # 清理当前层状态（有需要的话）
```
### 思维要点：
1. 不要人肉进行递归
2. 找到最近最简方法，将其拆解成可重复解决的问题（找最近重复子问题）
3. 数学归纳法思想
### 题目
* <span id="jump">[括号生成](https://leetcode-cn.com/problems/generate-parentheses/)</span>  
第一步，可以先考虑简单问题，即不考虑合法性时，怎么生成2*n大小的括号串：
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self._generate(0, 2 * n, s='')
        return


    def _generate(self, level, max_level, s):
        # 终止条件
        if level >= max_level:
            print(s)
            return

        # process current logic: join s with left bracket or right bracket
        # drill down
        self._generate(level + 1, max_level, s + '(')
        self._generate(level + 1, max_level, s + ')')
        # reverse state if needed
```
第二步，加入合法性判断：可以生成后用栈来判断；也可边生成边filter：1. 左括号：随时可以加只要不超标；2. 右括号：左括号个数>右括号个数才可加
```python
class Solution:
    def __init__(self):
        self.res = []

    def generateParenthesis(self, n: int) -> List[str]:
        self._generate(0, 0, n, '', [])
        return self.res


    def _generate(self, left, right, n, s, res):
        # 终止条件
        if left == n and right == n:
            self.res.append(s)
            return

        # process current logic: join s with left bracket or right bracket
        # drill down
        if left < n:
            self._generate(left + 1, right, n, s + '(', res)
        if right < left:
            self._generate(left, right + 1, n, s + ')', res)
        # reverse state if needed
```

## 二、分治与回溯
递归中的细分，本质是找问题中的重复性以及分解问题及组合每个子问题的结果  
### 分治模板：
```python
def divide_conquer(problem, param1, param2, ...):
    # recursion terminator
    if problem is None:
        print(result)
        return 
    # prepare data
    data = prepare_data(problem)
    subproblems = split_problem(problem, data)
    # conquer subproblems 
    subresult1 = self.divide_conquer(subproblems[0], p1, p2, ...)
    subresult2 = self.divide_conquer(subproblems[1], p1, p2, ...)
    ...
    # process and generate the final result
    result = process_result(subresult1, subresult2, ...)
    # revert current level states
```
### 题目
* [Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)  
思路1：暴力，乘n次，O(n)  
思路2：分治（牢记模板：1. terminator, 2. process (split), 3. drill down, merge 4. reverse)  
x^n --> (x^(n/2))*(x^(n/2))  O(logn)  
merge时要分奇偶
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def subPow(x, n):
            if n == 0:
                return 1
            half = subPow(x, n // 2)
            if n % 2 == 0 :
                return half * half
            else:
                return half * half * x
        
        if n < 0:
            n = -n
            return 1 / subPow(x, n)
        return subPow(x, n) 
```
* [子集](https://leetcode-cn.com/problems/subsets/)  
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
思路1. 和前面的[括号生成](#jump)思路类似，将数组看作不同层，[1, 2, 3]共三层，第一层选或不选，第二层选或不选，...。把选或不选写成递归的形式。  
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        if not nums:
            return ans
        self.dfs(ans, nums, [], 0)
        return ans
    
    def dfs(self, ans, nums, pre_list, index):
        if index == len(nums):
            ans.append(pre_list[:])
            return
        # 不加当前index
        self.dfs(ans, nums, pre_list, index + 1)
        # 加当前index
        pre_list.append(nums[index])
        self.dfs(ans, nums, pre_list, index + 1)
        # revert state
        pre_list.pop()
```
思路2. 迭代，向已有的集合的每一个元素新加上当前num，并归并形成新的集合
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for num in nums:
            res = res + [i + [num] for i in res]
        return res
```