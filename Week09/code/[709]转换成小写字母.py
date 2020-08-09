# 实现函数 ToLowerCase()，该函数接收一个字符串参数 str，并将该字符串中的大写字母转换成小写字母，之后返回新的字符串。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入: "Hello"
# 输出: "hello" 
# 
#  示例 2： 
# 
#  
# 输入: "here"
# 输出: "here" 
# 
#  示例 3： 
# 
#  
# 输入: "LOVELY"
# 输出: "lovely"
#  
#  Related Topics 字符串


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def toLowerCase(self, str: str) -> str:
        # if not str: return ''
        # new = ''
        # for s in str:
        #     if ord('A') <= ord(s) <= ord('Z'):
        #         new_c = chr(ord(s) - ord('A') + ord('a'))
        #         new += new_c
        #     else:
        #         new += s
        # return new

        # 位运算
        # 大写变小写、小写变大写： ^32
        # 大写变小写、小写变小写：| 32
        # 小写变大写、大写变大写： & -33
        if not str: return ''
        new = ''
        for s in str:
            new_c = chr(ord(s) | 32)
            new += new_c
        return new
# leetcode submit region end(Prohibit modification and deletion)
