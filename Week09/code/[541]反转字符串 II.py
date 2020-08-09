# 给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。 
# 
#  
#  如果剩余字符少于 k 个，则将剩余字符全部反转。 
#  如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。 
#  
# 
#  
# 
#  示例: 
# 
#  输入: s = "abcdefg", k = 2
# 输出: "bacdfeg"
#  
# 
#  
# 
#  提示： 
# 
#  
#  该字符串只包含小写英文字母。 
#  给定字符串的长度和 k 在 [1, 10000] 范围内。 
#  
#  Related Topics 字符串


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        if len(s) == 0: return s
        s = list(s)
        for i in range(0, len(s), 2*k):
            start, end = i, i + k - 1
            if i + k - 1 > len(s) - 1:
                end = len(s) - 1
            while start < end:
                s[start], s[end] = s[end], s[start]
                start += 1
                end -= 1
        return ''.join(s)

# leetcode submit region end(Prohibit modification and deletion)
