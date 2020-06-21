# 给定一个二叉树，返回它的中序 遍历。 
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
# 输出: [1,3,2] 
# 
#  进阶: 递归算法很简单，你可以通过迭代算法完成吗？ 
#  Related Topics 栈 树 哈希表


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 迭代
        res = []
        def dfs(root):
            if root:
                dfs(root.left)
                res.append(root.val)
                dfs(root.right)
        dfs(root)
        return res

        # 递归，颜色标记法
        # stack = [(0, root)]
        # res = []
        # while stack:
        #     color, node = stack.pop()
        #     if not node:
        #         continue
        #     if color == 0:
        #         stack.append((0, node.right))
        #         stack.append((1, node))
        #         stack.append((0, node.left))
        #     else:
        #         res.append(node.val)
        # return res

        # 递归
        # stack = []
        # res = []
        # while stack or root:
        #     if root:
        #         # 若存在左结点，一直往左边走
        #         stack.append(root)
        #         root = root.left
        #     else:
        #         # 如果左边已经做完，走右边
        #         node = stack.pop()
        #         res.append(node.val)
        #         root = node.right
        # return res
    # leetcode submit region end(Prohibit modification and deletion)
