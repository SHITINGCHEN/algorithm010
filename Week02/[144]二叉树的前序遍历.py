# 给定一个二叉树，返回它的 前序 遍历。 
# 
#  示例: 
# 
#  输入: [1,null,2,3]  
#    1
#     \
#      2
#     /
#    3 
# 
# 输出: [1,2,3]
#  
# 
#  进阶: 递归算法很简单，你可以通过迭代算法完成吗？ 
#  Related Topics 栈 树


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 迭代
        res = []
        def dfs(root):
            if root:
                res.append(root.val)
                dfs(root.left)
                dfs(root.right)
        dfs(root)
        return res

        # 颜色标记，递归
        # stack = [(0, root)]
        # res = []
        # while stack:
        #     color, node = stack.pop()
        #     if not node:
        #         continue
        #     if not color:
        #         stack.append((0, node.right))
        #         stack.append((0, node.left))
        #         stack.append((1, node))
        #     else:
        #         res.append(node.val)
        # return res

        # 递归
        # stack = [root]
        # res = []
        # while stack:
        #     node = stack.pop()
        #     if node:
        #         res.append(node.val)
        #         stack.append(node.right)
        #         stack.append(node.left)
        # return res

# leetcode submit region end(Prohibit modification and deletion)
