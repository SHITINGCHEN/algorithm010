# 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。 
# 
#  示例 1： 
# 
#  输入: "babad"
# 输出: "bab"
# 注意: "aba" 也是一个有效答案。
#  
# 
#  示例 2： 
# 
#  输入: "cbbd"
# 输出: "bb"
#  
#  Related Topics 字符串 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # if not s: return ''
        # max_substr = ''
        # n = len(s)
        # dp = [[False] * n for _ in range(n)]
        # for i in range(n - 1, -1, -1):
        #     for j in range(i, n):
        #         if j == i or (s[i] == s[j] and (dp[i + 1][j - 1] or j-i == 1)):
        #             dp[i][j] = True
        #             if j - i + 1 > len(max_substr):
        #                 max_substr = s[i:j+1]
        # return max_substr

        def expandFromCenter(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1

        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = expandFromCenter(s, i, i) # 奇数的子串
            left2, right2 = expandFromCenter(s, i, i + 1) # 偶数的子串

            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start:end+1]


# leetcode submit region end(Prohibit modification and deletion)
