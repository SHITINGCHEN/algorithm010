# 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。 
# 
#  说明：本题中，我们将空字符串定义为有效的回文串。 
# 
#  示例 1: 
# 
#  输入: "A man, a plan, a canal: Panama"
# 输出: true
#  
# 
#  示例 2: 
# 
#  输入: "race a car"
# 输出: false
#  
#  Related Topics 双指针 字符串


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isPalindrome(self, s: str) -> bool:
        def isDigitAlpha(char: str):
            return char.isdigit() or char.isalpha()

        if len(s) == 0: return True
        s = s.lower()
        left, right = 0, len(s) - 1
        while left < right:
            if isDigitAlpha(s[left]) and isDigitAlpha(s[right]):
                if s[left] != s[right]:
                    return False
                left += 1; right -= 1
            elif isDigitAlpha(s[left]):
                right -= 1
            else:
                left += 1
        return True


# leetcode submit region end(Prohibit modification and deletion)
